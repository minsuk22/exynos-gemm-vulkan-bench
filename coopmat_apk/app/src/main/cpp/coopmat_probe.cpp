// VK_KHR_cooperative_matrix probe + 2048^3 GEMM benchmark, one entry per
// advertised (A, B, C, Result, MxNxK, saturate) configuration.
//
// Flow:
//   1. pick the first physical device exposing VK_KHR_cooperative_matrix
//   2. dump every VkCooperativeMatrixPropertiesKHR it advertises
//   3. for each subgroup-scope configuration with a matching prebuilt kernel
//      (assets/shaders/gemm_<A>_<B>_<C>[_sat]_<v4|v1>.spv):
//        - short tuning sweep over workgroup/warp tiling at 2048^3
//        - final timed run of the best tiling (GPU timestamps)
//        - spot-check 64 output elements against a CPU reference
//   4. return everything as JSON; progress lines go to the Java callback and
//      logcat (tag COOPMAT)

#include <jni.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

#define TAG "COOPMAT"

namespace {

constexpr uint32_t kSize = 2048;   // M = N = K

// ---------------------------------------------------------------------------
// Enums and structs newer than the NDK r28 Vulkan header (VK_HEADER_VERSION 275)
// ---------------------------------------------------------------------------
constexpr VkComponentTypeKHR CT_BFLOAT16        = (VkComponentTypeKHR)1000141000; // VK_KHR_shader_bfloat16
constexpr VkComponentTypeKHR CT_SINT8_PACKED_NV = (VkComponentTypeKHR)1000491000; // VK_NV_cooperative_vector
constexpr VkComponentTypeKHR CT_UINT8_PACKED_NV = (VkComponentTypeKHR)1000491001;
constexpr VkComponentTypeKHR CT_FLOAT8_E4M3     = (VkComponentTypeKHR)1000491002; // VK_EXT_shader_float8
constexpr VkComponentTypeKHR CT_FLOAT8_E5M2     = (VkComponentTypeKHR)1000491003;

constexpr VkStructureType ST_BF16_FEATURES = (VkStructureType)1000141000;
constexpr VkStructureType ST_FP8_FEATURES  = (VkStructureType)1000567000;
constexpr const char *EXT_BF16 = "VK_KHR_shader_bfloat16";
constexpr const char *EXT_FP8  = "VK_EXT_shader_float8";

struct BFloat16Features {
    VkStructureType sType; void *pNext;
    VkBool32 shaderBFloat16Type, shaderBFloat16DotProduct, shaderBFloat16CooperativeMatrix;
};
struct Float8Features {
    VkStructureType sType; void *pNext;
    VkBool32 shaderFloat8, shaderFloat8CooperativeMatrix;
};

// ---------------------------------------------------------------------------
// Component types
// ---------------------------------------------------------------------------
enum class Kind { Float, Signed, Unsigned, Other };

struct TypeInfo {
    VkComponentTypeKHR t;
    const char *key;    // kernel file-name fragment
    const char *name;   // display name
    uint32_t bytes;
    Kind kind;
};

const TypeInfo kTypes[] = {
    { VK_COMPONENT_TYPE_FLOAT16_KHR, "f16",  "fp16",     2, Kind::Float },
    { VK_COMPONENT_TYPE_FLOAT32_KHR, "f32",  "fp32",     4, Kind::Float },
    { VK_COMPONENT_TYPE_FLOAT64_KHR, "f64",  "fp64",     8, Kind::Float },
    { CT_BFLOAT16,                   "bf16", "bf16",     2, Kind::Float },
    { CT_FLOAT8_E4M3,                "e4m3", "fp8_e4m3", 1, Kind::Float },
    { CT_FLOAT8_E5M2,                "e5m2", "fp8_e5m2", 1, Kind::Float },
    { VK_COMPONENT_TYPE_SINT8_KHR,   "s8",   "int8",     1, Kind::Signed },
    { VK_COMPONENT_TYPE_SINT16_KHR,  "s16",  "int16",    2, Kind::Signed },
    { VK_COMPONENT_TYPE_SINT32_KHR,  "s32",  "int32",    4, Kind::Signed },
    { VK_COMPONENT_TYPE_SINT64_KHR,  "s64",  "int64",    8, Kind::Signed },
    { VK_COMPONENT_TYPE_UINT8_KHR,   "u8",   "uint8",    1, Kind::Unsigned },
    { VK_COMPONENT_TYPE_UINT16_KHR,  "u16",  "uint16",   2, Kind::Unsigned },
    { VK_COMPONENT_TYPE_UINT32_KHR,  "u32",  "uint32",   4, Kind::Unsigned },
    { VK_COMPONENT_TYPE_UINT64_KHR,  "u64",  "uint64",   8, Kind::Unsigned },
    { CT_SINT8_PACKED_NV,            "s8p",  "int8x4_packed",  4, Kind::Other },
    { CT_UINT8_PACKED_NV,            "u8p",  "uint8x4_packed", 4, Kind::Other },
};

const TypeInfo *typeInfo(VkComponentTypeKHR t) {
    for (const auto &ti : kTypes) if (ti.t == t) return &ti;
    return nullptr;
}

std::string typeName(VkComponentTypeKHR t) {
    const TypeInfo *ti = typeInfo(t);
    return ti ? ti->name : "type#" + std::to_string((int)t);
}

const char *scopeName(VkScopeKHR s) {
    switch (s) {
        case VK_SCOPE_DEVICE_KHR:       return "device";
        case VK_SCOPE_WORKGROUP_KHR:    return "workgroup";
        case VK_SCOPE_SUBGROUP_KHR:     return "subgroup";
        case VK_SCOPE_QUEUE_FAMILY_KHR: return "queue_family";
        default:                        return "unknown";
    }
}

std::string stageNames(VkShaderStageFlags f) {
    std::string s;
    auto add = [&](VkShaderStageFlags b, const char *n) {
        if (f & b) { if (!s.empty()) s += "|"; s += n; }
    };
    add(VK_SHADER_STAGE_COMPUTE_BIT, "compute");
    add(VK_SHADER_STAGE_VERTEX_BIT, "vertex");
    add(VK_SHADER_STAGE_FRAGMENT_BIT, "fragment");
    add(VK_SHADER_STAGE_GEOMETRY_BIT, "geometry");
    add(VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, "tess_ctrl");
    add(VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, "tess_eval");
    add(VK_SHADER_STAGE_TASK_BIT_EXT, "task");
    add(VK_SHADER_STAGE_MESH_BIT_EXT, "mesh");
    add(VK_SHADER_STAGE_RAYGEN_BIT_KHR, "raygen");
    return s.empty() ? "none" : s;
}

// ---------------------------------------------------------------------------
// Value encoding. Inputs are drawn from {-1,0,1} (signed / float types) or
// {0,1,2} (unsigned), so every format represents them exactly and every
// partial sum of a 2048-long dot product is a small integer.
// ---------------------------------------------------------------------------
uint32_t floatBits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
float bitsFloat(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

float halfToFloat(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    float v;
    if (e == 0)       v = std::ldexp((float)m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else              v = std::ldexp((float)(m | 0x400), (int)e - 25);
    return s ? -v : v;
}

uint32_t encodeSmall(VkComponentTypeKHR t, int v) {
    switch ((int)t) {
        case VK_COMPONENT_TYPE_FLOAT16_KHR:
            return v == 0 ? 0 : v == 1 ? 0x3C00 : v == -1 ? 0xBC00 : 0x4000;
        case VK_COMPONENT_TYPE_FLOAT32_KHR: return floatBits((float)v);
        case (int)CT_BFLOAT16:              return floatBits((float)v) >> 16;
        case (int)CT_FLOAT8_E4M3:           // bias 7, 3 mantissa bits
            return v == 0 ? 0 : v == 1 ? 0x38 : v == -1 ? 0xB8 : 0x40;
        case (int)CT_FLOAT8_E5M2:           // bias 15, 2 mantissa bits
            return v == 0 ? 0 : v == 1 ? 0x3C : v == -1 ? 0xBC : 0x40;
        default:                            return (uint32_t)v;  // integers, truncated by store width
    }
}

double decodeOut(VkComponentTypeKHR t, const void *base, size_t idx) {
    switch ((int)t) {
        case VK_COMPONENT_TYPE_FLOAT16_KHR: return halfToFloat(((const uint16_t *)base)[idx]);
        case (int)CT_BFLOAT16:              return bitsFloat((uint32_t)((const uint16_t *)base)[idx] << 16);
        case VK_COMPONENT_TYPE_FLOAT32_KHR: return ((const float *)base)[idx];
        case VK_COMPONENT_TYPE_SINT32_KHR:  return ((const int32_t *)base)[idx];
        case VK_COMPONENT_TYPE_UINT32_KHR:  return ((const uint32_t *)base)[idx];
        case VK_COMPONENT_TYPE_SINT16_KHR:  return ((const int16_t *)base)[idx];
        case VK_COMPONENT_TYPE_UINT16_KHR:  return ((const uint16_t *)base)[idx];
        default:                            return NAN;
    }
}

struct Lcg {
    uint32_t s;
    explicit Lcg(uint32_t seed) : s(seed) {}
    uint32_t next() { s = s * 1664525u + 1013904223u; return s >> 8; }
};

// ---------------------------------------------------------------------------
// Tiny JSON writer
// ---------------------------------------------------------------------------
class Json {
public:
    Json &obj(const char *k = nullptr) { key(k); out_ += '{'; first_.push_back(true); return *this; }
    Json &arr(const char *k = nullptr) { key(k); out_ += '['; first_.push_back(true); return *this; }
    Json &endObj() { out_ += '}'; first_.pop_back(); return *this; }
    Json &endArr() { out_ += ']'; first_.pop_back(); return *this; }
    Json &kv(const char *k, const std::string &v) { key(k); str(v); return *this; }
    Json &kv(const char *k, const char *v) { return kv(k, std::string(v)); }
    Json &kv(const char *k, bool v) { key(k); out_ += v ? "true" : "false"; return *this; }
    Json &kv(const char *k, int v) { return kv(k, (int64_t)v); }
    Json &kv(const char *k, uint32_t v) { return kv(k, (int64_t)v); }
    Json &kv(const char *k, int64_t v) { key(k); out_ += std::to_string(v); return *this; }
    Json &kv(const char *k, double v) {
        key(k);
        if (!std::isfinite(v)) { out_ += "null"; return *this; }
        char b[64]; snprintf(b, sizeof b, "%.6g", v); out_ += b; return *this;
    }
    const std::string &str() const { return out_; }

private:
    void key(const char *k) {
        if (!first_.empty()) {
            if (!first_.back()) out_ += ',';
            first_.back() = false;
        }
        if (k) { str(k); out_ += ':'; }
    }
    void str(const std::string &s) {
        out_ += '"';
        for (char c : s) {
            switch (c) {
                case '"':  out_ += "\\\""; break;
                case '\\': out_ += "\\\\"; break;
                case '\n': out_ += "\\n"; break;
                case '\t': out_ += "\\t"; break;
                default:
                    if ((unsigned char)c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); out_ += b; }
                    else out_ += c;
            }
        }
        out_ += '"';
    }
    std::string out_;
    std::vector<bool> first_;
};

// ---------------------------------------------------------------------------
// Progress / log sink
// ---------------------------------------------------------------------------
struct Log {
    JNIEnv *env = nullptr;
    jobject cb = nullptr;
    jmethodID mid = nullptr;
    std::string text;

    void line(const char *fmt, ...) __attribute__((format(printf, 2, 3))) {
        char buf[1024];
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(buf, sizeof buf, fmt, ap);
        va_end(ap);
        text += buf;
        text += '\n';
        __android_log_print(ANDROID_LOG_INFO, TAG, "%s", buf);
        if (env && cb && mid) {
            jstring s = env->NewStringUTF(buf);
            env->CallVoidMethod(cb, mid, s);
            env->DeleteLocalRef(s);
            if (env->ExceptionCheck()) env->ExceptionClear();
        }
    }
};

// ---------------------------------------------------------------------------
// Vulkan context
// ---------------------------------------------------------------------------
struct Features {
    bool coopmat = false, vmm = false, f16 = false, i8 = false, i16 = false, storage16 = false;
    bool sgsc = false, fullSubgroups = false;
    bool bf16Ext = false, bf16Type = false, bf16Coop = false;
    bool fp8Ext = false, fp8 = false, fp8Coop = false;
};

struct Gpu {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t qfi = 0;
    VkPhysicalDeviceProperties props{};
    VkPhysicalDeviceMemoryProperties mem{};
    Features f;
    uint32_t sgSize = 0, sgMin = 0, sgMax = 0;
    VkShaderStageFlags sgReqStages = 0;
    VkShaderStageFlags coopStages = 0;
    uint32_t tsBits = 0;
    std::string driverName, driverInfo;

    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkQueryPool qpool = VK_NULL_HANDLE;
    bool lost = false;
};

constexpr uint32_t kMaxTimed = 32;

struct Buf {
    VkBuffer b = VK_NULL_HANDLE;
    VkDeviceMemory m = VK_NULL_HANDLE;
    void *p = nullptr;
    VkDeviceSize size = 0;
};

bool makeBuf(Gpu &g, VkDeviceSize size, Buf &out) {
    VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(g.dev, &bci, nullptr, &out.b) != VK_SUCCESS) return false;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(g.dev, out.b, &req);

    const VkMemoryPropertyFlags want[] = {
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    };
    int type = -1;
    for (VkMemoryPropertyFlags w : want) {
        for (uint32_t i = 0; i < g.mem.memoryTypeCount && type < 0; ++i)
            if ((req.memoryTypeBits & (1u << i)) && (g.mem.memoryTypes[i].propertyFlags & w) == w)
                type = (int)i;
        if (type >= 0) break;
    }
    if (type < 0) return false;

    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = (uint32_t)type;
    if (vkAllocateMemory(g.dev, &mai, nullptr, &out.m) != VK_SUCCESS) return false;
    vkBindBufferMemory(g.dev, out.b, out.m, 0);
    if (vkMapMemory(g.dev, out.m, 0, VK_WHOLE_SIZE, 0, &out.p) != VK_SUCCESS) return false;
    out.size = size;
    return true;
}

void freeBuf(Gpu &g, Buf &b) {
    if (b.m) { if (b.p) vkUnmapMemory(g.dev, b.m); vkFreeMemory(g.dev, b.m, nullptr); }
    if (b.b) vkDestroyBuffer(g.dev, b.b, nullptr);
    b = Buf{};
}

bool hasExt(const std::vector<VkExtensionProperties> &exts, const char *name) {
    for (const auto &e : exts) if (strcmp(e.extensionName, name) == 0) return true;
    return false;
}

// ---------------------------------------------------------------------------
// Benchmark plumbing
// ---------------------------------------------------------------------------
struct Tile {
    uint32_t cRows, cCols, wgW, wgH, bk;
    uint32_t tileM(uint32_t lM) const { return wgH * cRows * lM; }
    uint32_t tileN(uint32_t lN) const { return wgW * cCols * lN; }
};

std::string tileDesc(const Tile &t, uint32_t lM, uint32_t lN, uint32_t sg) {
    char b[160];
    snprintf(b, sizeof b, "WG %ux%u (%u sg x %u = %u thr), warp %ux%u frags, block %ux%u, BK %u",
             t.wgW, t.wgH, t.wgW * t.wgH, sg, t.wgW * t.wgH * sg,
             t.cRows, t.cCols, t.tileM(lM), t.tileN(lN), t.bk);
    return b;
}

VkPipeline makePipeline(Gpu &g, VkShaderModule mod, uint32_t lM, uint32_t lN, uint32_t lK,
                        const Tile &t, VkResult &err) {
    uint32_t local = t.wgW * t.wgH * g.sgSize;
    uint32_t spec[12] = { lM, lN, lK, kSize, kSize, kSize,
                          t.cRows, t.cCols, t.wgW, t.wgH, t.bk, local };
    VkSpecializationMapEntry ent[12];
    for (uint32_t i = 0; i < 12; ++i) ent[i] = { i, i * 4, 4 };
    VkSpecializationInfo si{ 12, ent, sizeof spec, spec };

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rss{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO };
    rss.requiredSubgroupSize = g.sgSize;

    VkComputePipelineCreateInfo cpci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = mod;
    cpci.stage.pName = "main";
    cpci.stage.pSpecializationInfo = &si;
    if (g.f.sgsc && (g.sgReqStages & VK_SHADER_STAGE_COMPUTE_BIT)) cpci.stage.pNext = &rss;
    if (g.f.fullSubgroups) cpci.stage.flags |= VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
    cpci.layout = g.layout;

    VkPipeline p = VK_NULL_HANDLE;
    err = vkCreateComputePipelines(g.dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &p);
    if (err != VK_SUCCESS) p = VK_NULL_HANDLE;
    return p;
}

// Runs `warmup` untimed + `iters` timed dispatches in one submit. Fills ms[]
// with per-dispatch GPU time (consecutive bottom-of-pipe timestamps).
bool runTimed(Gpu &g, VkPipeline pipe, uint32_t gx, uint32_t gy, uint32_t warmup, uint32_t iters,
              std::vector<double> &ms, std::string &err) {
    iters = std::min(iters, kMaxTimed - 1);
    bool useTs = g.tsBits > 0 && g.qpool;

    vkResetCommandBuffer(g.cmd, 0);
    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(g.cmd, &bi);
    if (useTs) vkCmdResetQueryPool(g.cmd, g.qpool, 0, kMaxTimed);
    vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.layout, 0, 1, &g.set, 0, nullptr);

    VkMemoryBarrier mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    auto barrier = [&] {
        vkCmdPipelineBarrier(g.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &mb, 0, nullptr, 0, nullptr);
    };
    for (uint32_t i = 0; i < warmup; ++i) { vkCmdDispatch(g.cmd, gx, gy, 1); barrier(); }
    if (useTs) vkCmdWriteTimestamp(g.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g.qpool, 0);
    for (uint32_t i = 0; i < iters; ++i) {
        vkCmdDispatch(g.cmd, gx, gy, 1);
        if (useTs) vkCmdWriteTimestamp(g.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g.qpool, i + 1);
        barrier();
    }
    VkMemoryBarrier hb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    hb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    hb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(g.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                         0, 1, &hb, 0, nullptr, 0, nullptr);
    vkEndCommandBuffer(g.cmd);

    vkResetFences(g.dev, 1, &g.fence);
    VkSubmitInfo sub{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    sub.commandBufferCount = 1;
    sub.pCommandBuffers = &g.cmd;
    auto t0 = std::chrono::steady_clock::now();
    VkResult r = vkQueueSubmit(g.queue, 1, &sub, g.fence);
    if (r != VK_SUCCESS) { err = "vkQueueSubmit=" + std::to_string(r); if (r == VK_ERROR_DEVICE_LOST) g.lost = true; return false; }
    r = vkWaitForFences(g.dev, 1, &g.fence, VK_TRUE, 30ull * 1000 * 1000 * 1000);
    auto t1 = std::chrono::steady_clock::now();
    if (r != VK_SUCCESS) {
        err = r == VK_TIMEOUT ? "GPU timeout (>30 s)" : "vkWaitForFences=" + std::to_string(r);
        g.lost = true;   // a hung queue is not recoverable here
        return false;
    }

    ms.clear();
    if (useTs) {
        uint64_t ts[kMaxTimed];
        r = vkGetQueryPoolResults(g.dev, g.qpool, 0, iters + 1, sizeof ts, ts, sizeof(uint64_t),
                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        if (r == VK_SUCCESS) {
            uint64_t mask = g.tsBits >= 64 ? ~0ull : ((1ull << g.tsBits) - 1);
            for (uint32_t i = 0; i < iters; ++i)
                ms.push_back((double)((ts[i + 1] - ts[i]) & mask) * g.props.limits.timestampPeriod * 1e-6);
            return true;
        }
    }
    double total = std::chrono::duration<double, std::milli>(t1 - t0).count();
    for (uint32_t i = 0; i < iters; ++i) ms.push_back(total / (warmup + iters));
    return true;
}

double median(std::vector<double> v) {
    if (v.empty()) return NAN;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

double tops(double ms) { return 2.0 * kSize * kSize * (double)kSize / (ms * 1e-3) / 1e12; }

std::vector<uint8_t> loadAsset(AAssetManager *am, const std::string &path) {
    std::vector<uint8_t> v;
    if (!am) return v;
    AAsset *a = AAssetManager_open(am, path.c_str(), AASSET_MODE_BUFFER);
    if (!a) return v;
    size_t n = AAsset_getLength(a);
    v.resize(n);
    memcpy(v.data(), AAsset_getBuffer(a), n);
    AAsset_close(a);
    return v;
}

// ---------------------------------------------------------------------------
// Bench entry
// ---------------------------------------------------------------------------
struct TuneResult { Tile t; double tops; std::string note; };

struct Entry {
    VkCooperativeMatrixPropertiesKHR cfg;
    std::string status = "skip";   // ok | skip | fail
    std::string reason;
    std::string kernel;
    std::string tileText;
    double ms = NAN, msBest = NAN, value = NAN, valueBest = NAN;
    std::string check;
    double maxErr = NAN;
    std::vector<TuneResult> tune;
};

// Reasons this configuration cannot be benchmarked by our kernel/features.
std::string unsupportedReason(const Gpu &g, const VkCooperativeMatrixPropertiesKHR &c) {
    if (c.CType != c.ResultType) return "C type != Result type (kernel accumulates in place)";
    for (VkComponentTypeKHR t : { c.AType, c.BType, c.CType }) {
        const TypeInfo *ti = typeInfo(t);
        if (!ti) return "unknown component type " + std::to_string((int)t);
        switch ((int)t) {
            case VK_COMPONENT_TYPE_FLOAT16_KHR: if (!g.f.f16) return "shaderFloat16 not supported"; break;
            case (int)CT_BFLOAT16:
                if (!g.f.bf16Coop) return "shaderBFloat16CooperativeMatrix not supported"; break;
            case (int)CT_FLOAT8_E4M3: case (int)CT_FLOAT8_E5M2:
                if (!g.f.fp8Coop) return "shaderFloat8CooperativeMatrix not supported"; break;
            case VK_COMPONENT_TYPE_SINT8_KHR: case VK_COMPONENT_TYPE_UINT8_KHR:
                if (!g.f.i8) return "shaderInt8 not supported"; break;
            case VK_COMPONENT_TYPE_SINT16_KHR: case VK_COMPONENT_TYPE_UINT16_KHR:
                if (!g.f.i16) return "shaderInt16 not supported"; break;
            case VK_COMPONENT_TYPE_FLOAT32_KHR: case VK_COMPONENT_TYPE_SINT32_KHR:
            case VK_COMPONENT_TYPE_UINT32_KHR: break;
            default: return std::string("no kernel for ") + ti->name;
        }
    }
    if (typeInfo(c.CType)->bytes == 2 && !g.f.storage16)
        return "storageBuffer16BitAccess not supported (16-bit result buffer)";
    if (kSize % c.MSize || kSize % c.NSize || kSize % c.KSize)
        return "shape does not divide 2048";
    return "";
}

void benchEntry(Gpu &g, AAssetManager *am, Log &log, Entry &e) {
    const auto &c = e.cfg;
    const TypeInfo &A = *typeInfo(c.AType), &B = *typeInfo(c.BType), &C = *typeInfo(c.CType);
    const uint32_t lM = c.MSize, lN = c.NSize, lK = c.KSize;

    // Pick the widest staging unit the fragment offsets stay aligned to.
    uint32_t ld = 0;
    for (uint32_t cand : { 16u, 4u })
        if ((lK * A.bytes) % cand == 0 && (lN * B.bytes) % cand == 0) { ld = cand; break; }
    if (!ld) { e.reason = "fragment row not 4-byte aligned"; return; }

    std::string base = std::string("gemm_") + A.key + "_" + B.key + "_" + C.key + (c.saturatingAccumulation ? "_sat" : "");
    std::string file = base + (ld == 16 ? "_v4" : "_v1") + ".spv";
    std::vector<uint8_t> spv = loadAsset(am, "shaders/" + file);
    if (spv.empty()) { e.reason = "no prebuilt kernel (" + file + ")"; return; }
    e.kernel = file;

    VkShaderModuleCreateInfo smci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = spv.size();
    smci.pCode = (const uint32_t *)spv.data();
    VkShaderModule mod;
    VkResult r = vkCreateShaderModule(g.dev, &smci, nullptr, &mod);
    if (r != VK_SUCCESS) { e.status = "fail"; e.reason = "vkCreateShaderModule=" + std::to_string(r); return; }

    // Buffers + data.
    const size_t elems = (size_t)kSize * kSize;
    Buf bA, bB, bD;
    if (!makeBuf(g, elems * A.bytes, bA) || !makeBuf(g, elems * B.bytes, bB) || !makeBuf(g, elems * C.bytes, bD)) {
        e.status = "fail"; e.reason = "buffer allocation failed";
        freeBuf(g, bA); freeBuf(g, bB); freeBuf(g, bD);
        vkDestroyShaderModule(g.dev, mod, nullptr);
        return;
    }
    std::vector<int8_t> hA(elems), hB(elems);
    auto fill = [&](const TypeInfo &T, std::vector<int8_t> &h, Buf &buf, uint32_t seed) {
        Lcg rng(seed);
        uint32_t lut[3];
        int vals[3];
        for (int i = 0; i < 3; ++i) {
            vals[i] = T.kind == Kind::Unsigned ? i : i - 1;
            lut[i] = encodeSmall(T.t, vals[i]);
        }
        for (size_t i = 0; i < elems; ++i) {
            int s = (int)(rng.next() % 3);
            h[i] = (int8_t)vals[s];
            switch (T.bytes) {
                case 1: ((uint8_t *)buf.p)[i] = (uint8_t)lut[s]; break;
                case 2: ((uint16_t *)buf.p)[i] = (uint16_t)lut[s]; break;
                default: ((uint32_t *)buf.p)[i] = lut[s]; break;
            }
        }
    };
    fill(A, hA, bA, 0x1234u);
    fill(B, hB, bB, 0xBEEFu);
    memset(bD.p, 0, bD.size);

    VkDescriptorBufferInfo dbi[3] = { { bA.b, 0, VK_WHOLE_SIZE }, { bB.b, 0, VK_WHOLE_SIZE }, { bD.b, 0, VK_WHOLE_SIZE } };
    VkWriteDescriptorSet w[3];
    for (uint32_t i = 0; i < 3; ++i) {
        w[i] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w[i].dstSet = g.set;
        w[i].dstBinding = i;
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(g.dev, 3, w, 0, nullptr);

    // Tiling candidates. BK0 = smallest multiple of lK spanning >= 32 bytes of A.
    uint32_t bk0 = lK;
    while (bk0 * A.bytes < 32) bk0 += lK;
    const uint32_t shapes[][4] = {   // cRows, cCols, wgW, wgH
        { 4, 2, 4, 2 }, { 2, 2, 4, 2 }, { 2, 4, 2, 4 },
        { 4, 4, 2, 2 }, { 4, 2, 2, 2 }, { 2, 2, 2, 2 },
    };
    std::vector<Tile> cands;
    for (const auto &s : shapes)
        for (uint32_t bk : { bk0, bk0 * 2 }) cands.push_back({ s[0], s[1], s[2], s[3], bk });

    const auto &L = g.props.limits;
    Tile best{};
    double bestTops = -1;
    for (const Tile &t : cands) {
        if (g.lost) break;
        TuneResult tr{ t, NAN, "" };
        uint32_t TM = t.tileM(lM), TN = t.tileN(lN), thr = t.wgW * t.wgH * g.sgSize;
        uint32_t lds = (TM * (t.bk * A.bytes / ld + 1) + t.bk * (TN * B.bytes / ld + 1)) * ld;
        if (kSize % TM || kSize % TN || kSize % t.bk || (t.bk * A.bytes) % ld) tr.note = "tile does not divide 2048";
        else if (lds > L.maxComputeSharedMemorySize) tr.note = "LDS " + std::to_string(lds) + " B > limit";
        else if (thr > L.maxComputeWorkGroupInvocations || thr > L.maxComputeWorkGroupSize[0]) tr.note = "too many invocations";
        if (tr.note.empty()) {
            VkResult pr;
            VkPipeline p = makePipeline(g, mod, lM, lN, lK, t, pr);
            if (!p) tr.note = "pipeline creation failed (" + std::to_string(pr) + ")";
            else {
                std::vector<double> ms;
                std::string err;
                if (runTimed(g, p, kSize / TN, kSize / TM, 1, 3, ms, err)) {
                    tr.tops = tops(median(ms));
                    if (tr.tops > bestTops) { bestTops = tr.tops; best = t; }
                } else tr.note = err;
                vkDestroyPipeline(g.dev, p, nullptr);
            }
        }
        e.tune.push_back(tr);
    }

    if (bestTops <= 0) {
        e.status = "fail";
        e.reason = "no tiling ran";
        for (const auto &tr : e.tune) if (!tr.note.empty()) { e.reason += ": " + tr.note; break; }
    } else {
        VkResult pr;
        VkPipeline p = makePipeline(g, mod, lM, lN, lK, best, pr);
        std::vector<double> ms;
        std::string err;
        uint32_t TM = best.tileM(lM), TN = best.tileN(lN);
        if (p && runTimed(g, p, kSize / TN, kSize / TM, 2, 10, ms, err)) {
            e.ms = median(ms);
            e.msBest = *std::min_element(ms.begin(), ms.end());
            e.value = tops(e.ms);
            e.valueBest = tops(e.msBest);
            e.tileText = tileDesc(best, lM, lN, g.sgSize);

            // Spot-check 64 outputs (corners + random).
            Lcg rng(0xC0FFEEu);
            int bad = 0;
            double maxErr = 0;
            bool intOut = C.kind != Kind::Float;
            for (int s = 0; s < 64; ++s) {
                uint32_t row, col;
                if (s == 0) { row = 0; col = 0; }
                else if (s == 1) { row = kSize - 1; col = kSize - 1; }
                else { row = rng.next() % kSize; col = rng.next() % kSize; }
                int64_t ref = 0;
                for (uint32_t k = 0; k < kSize; ++k)
                    ref += (int64_t)hA[(size_t)row * kSize + k] * hB[(size_t)k * kSize + col];
                double got = decodeOut(C.t, bD.p, (size_t)row * kSize + col);
                double errv = std::fabs(got - (double)ref);
                double tol = intOut || C.t == VK_COMPONENT_TYPE_FLOAT32_KHR ? 0.0
                           : C.t == VK_COMPONENT_TYPE_FLOAT16_KHR ? 1.0
                           : std::max(2.0, 0.02 * std::fabs((double)ref));   // bf16 accumulator
                if (!(errv <= tol)) ++bad;
                if (std::isfinite(errv)) maxErr = std::max(maxErr, errv); else maxErr = INFINITY;
            }
            e.maxErr = maxErr;
            e.check = bad ? "FAIL " + std::to_string(bad) + "/64" : "pass 64/64";
            e.status = bad ? "fail" : "ok";
            if (bad) e.reason = "wrong results";
        } else {
            e.status = "fail";
            e.reason = p ? err : "pipeline creation failed (" + std::to_string(pr) + ")";
        }
        if (p) vkDestroyPipeline(g.dev, p, nullptr);
    }

    freeBuf(g, bA); freeBuf(g, bB); freeBuf(g, bD);
    vkDestroyShaderModule(g.dev, mod, nullptr);
}

// ---------------------------------------------------------------------------
// Top level
// ---------------------------------------------------------------------------
std::string run(AAssetManager *am, Log &log) {
    Json j;
    j.obj();
    j.kv("problem", (int)kSize);
    Gpu g;
    std::vector<Entry> entries;
    std::vector<VkCooperativeMatrixPropertiesKHR> coop;
    std::string fatal;

    auto finish = [&]() -> std::string {
        if (!fatal.empty()) j.kv("error", fatal);
        j.kv("log", log.text);
        j.endObj();
        if (g.dev) {
            vkDeviceWaitIdle(g.dev);
            if (g.qpool) vkDestroyQueryPool(g.dev, g.qpool, nullptr);
            if (g.fence) vkDestroyFence(g.dev, g.fence, nullptr);
            if (g.cmdPool) vkDestroyCommandPool(g.dev, g.cmdPool, nullptr);
            if (g.pool) vkDestroyDescriptorPool(g.dev, g.pool, nullptr);
            if (g.layout) vkDestroyPipelineLayout(g.dev, g.layout, nullptr);
            if (g.dsl) vkDestroyDescriptorSetLayout(g.dev, g.dsl, nullptr);
            vkDestroyDevice(g.dev, nullptr);
        }
        if (g.instance) vkDestroyInstance(g.instance, nullptr);
        return j.str();
    };

    // --- instance ---
    VkApplicationInfo app{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.pApplicationName = "CoopMatProbe";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &app;
    VkResult r = vkCreateInstance(&ici, nullptr, &g.instance);
    if (r != VK_SUCCESS) { app.apiVersion = VK_API_VERSION_1_1; r = vkCreateInstance(&ici, nullptr, &g.instance); }
    if (r != VK_SUCCESS) { g.instance = VK_NULL_HANDLE; fatal = "vkCreateInstance failed (" + std::to_string(r) + ")"; log.line("%s", fatal.c_str()); return finish(); }

    // --- physical device ---
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(g.instance, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    vkEnumeratePhysicalDevices(g.instance, &n, pds.data());
    std::vector<VkExtensionProperties> exts;
    for (VkPhysicalDevice pd : pds) {
        uint32_t ne = 0;
        vkEnumerateDeviceExtensionProperties(pd, nullptr, &ne, nullptr);
        std::vector<VkExtensionProperties> ex(ne);
        vkEnumerateDeviceExtensionProperties(pd, nullptr, &ne, ex.data());
        VkPhysicalDeviceProperties pp;
        vkGetPhysicalDeviceProperties(pd, &pp);
        bool has = hasExt(ex, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
        log.line("GPU: %s  (Vulkan %u.%u.%u)  VK_KHR_cooperative_matrix: %s", pp.deviceName,
                 VK_API_VERSION_MAJOR(pp.apiVersion), VK_API_VERSION_MINOR(pp.apiVersion),
                 VK_API_VERSION_PATCH(pp.apiVersion), has ? "yes" : "NO");
        if (!g.pd || (has && !hasExt(exts, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME))) { g.pd = pd; exts = ex; }
    }
    if (!g.pd) { fatal = "no Vulkan device"; return finish(); }
    vkGetPhysicalDeviceProperties(g.pd, &g.props);
    vkGetPhysicalDeviceMemoryProperties(g.pd, &g.mem);
    const bool hasCoop = hasExt(exts, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    const uint32_t api = g.props.apiVersion;
    g.f.bf16Ext = hasExt(exts, EXT_BF16);
    g.f.fp8Ext = hasExt(exts, EXT_FP8);
    const bool hasSgscExt = api >= VK_API_VERSION_1_3 || hasExt(exts, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
    const bool hasDriverProps = api >= VK_API_VERSION_1_2 || hasExt(exts, VK_KHR_DRIVER_PROPERTIES_EXTENSION_NAME);

    // --- properties ---
    VkPhysicalDeviceSubgroupProperties sgp{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES };
    VkPhysicalDeviceSubgroupSizeControlProperties sgscp{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES };
    VkPhysicalDeviceDriverProperties drv{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES };
    VkPhysicalDeviceCooperativeMatrixPropertiesKHR cmp{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR };
    VkPhysicalDeviceProperties2 p2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
    void **tail = &p2.pNext;
    auto chain = [&](void *s, void **next) { *tail = s; tail = next; };
    chain(&sgp, &sgp.pNext);
    if (hasSgscExt) chain(&sgscp, &sgscp.pNext);
    if (hasDriverProps) chain(&drv, &drv.pNext);
    if (hasCoop) chain(&cmp, &cmp.pNext);
    vkGetPhysicalDeviceProperties2(g.pd, &p2);
    g.sgSize = sgp.subgroupSize;
    g.sgMin = sgscp.minSubgroupSize;
    g.sgMax = sgscp.maxSubgroupSize;
    g.sgReqStages = sgscp.requiredSubgroupSizeStages;
    g.coopStages = cmp.cooperativeMatrixSupportedStages;
    if (hasDriverProps) { g.driverName = drv.driverName; g.driverInfo = drv.driverInfo; }

    // --- features ---
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR fc{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR };
    VkPhysicalDeviceVulkanMemoryModelFeatures fv{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES };
    VkPhysicalDeviceShaderFloat16Int8Features ff{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES };
    VkPhysicalDevice16BitStorageFeatures fs{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES };
    VkPhysicalDeviceSubgroupSizeControlFeatures fg{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES };
    BFloat16Features fb{ ST_BF16_FEATURES, nullptr, 0, 0, 0 };
    Float8Features f8{ ST_FP8_FEATURES, nullptr, 0, 0 };
    VkPhysicalDeviceFeatures2 f2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    tail = &f2.pNext;
    if (hasCoop) chain(&fc, &fc.pNext);
    chain(&fv, &fv.pNext);
    chain(&ff, &ff.pNext);
    chain(&fs, &fs.pNext);
    if (hasSgscExt) chain(&fg, &fg.pNext);
    if (g.f.bf16Ext) chain(&fb, &fb.pNext);
    if (g.f.fp8Ext) chain(&f8, &f8.pNext);
    vkGetPhysicalDeviceFeatures2(g.pd, &f2);
    g.f.coopmat = fc.cooperativeMatrix;
    g.f.vmm = fv.vulkanMemoryModel;
    g.f.f16 = ff.shaderFloat16;
    g.f.i8 = ff.shaderInt8;
    g.f.i16 = f2.features.shaderInt16;
    g.f.storage16 = fs.storageBuffer16BitAccess;
    g.f.sgsc = fg.subgroupSizeControl;
    g.f.fullSubgroups = fg.computeFullSubgroups;
    g.f.bf16Type = fb.shaderBFloat16Type;
    g.f.bf16Coop = fb.shaderBFloat16Type && fb.shaderBFloat16CooperativeMatrix;
    g.f.fp8 = f8.shaderFloat8;
    g.f.fp8Coop = f8.shaderFloat8 && f8.shaderFloat8CooperativeMatrix;

    // --- coopmat configurations ---
    if (hasCoop) {
        auto pfn = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
            vkGetInstanceProcAddr(g.instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
        if (pfn) {
            uint32_t nc = 0;
            pfn(g.pd, &nc, nullptr);
            coop.resize(nc);
            for (auto &c : coop) { c = {}; c.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR; }
            pfn(g.pd, &nc, coop.data());
            coop.resize(nc);
        }
    }

    log.line("");
    log.line("Device : %s", g.props.deviceName);
    log.line("Driver : %s %s", g.driverName.c_str(), g.driverInfo.c_str());
    log.line("Subgroup size %u (range %u..%u), maxSharedMem %u B, maxInvocations %u",
             g.sgSize, g.sgMin, g.sgMax, g.props.limits.maxComputeSharedMemorySize,
             g.props.limits.maxComputeWorkGroupInvocations);
    log.line("Features: coopmat=%d vmm=%d f16=%d i8=%d i16=%d storage16=%d sgsc=%d fullSg=%d",
             g.f.coopmat, g.f.vmm, g.f.f16, g.f.i8, g.f.i16, g.f.storage16, g.f.sgsc, g.f.fullSubgroups);
    log.line("          bf16: ext=%d type=%d coopmat=%d   fp8: ext=%d type=%d coopmat=%d",
             g.f.bf16Ext, fb.shaderBFloat16Type, fb.shaderBFloat16CooperativeMatrix,
             g.f.fp8Ext, f8.shaderFloat8, f8.shaderFloat8CooperativeMatrix);
    log.line("coopmat supported stages: %s", stageNames(g.coopStages).c_str());
    log.line("");
    log.line("Advertised cooperative matrix configurations: %zu", coop.size());
    for (const auto &c : coop)
        log.line("  %-9s A=%-8s B=%-8s C=%-8s R=%-8s %3ux%-3ux%-3u sat=%u",
                 scopeName(c.scope), typeName(c.AType).c_str(), typeName(c.BType).c_str(),
                 typeName(c.CType).c_str(), typeName(c.ResultType).c_str(),
                 c.MSize, c.NSize, c.KSize, c.saturatingAccumulation);

    // --- JSON: device / features / configs ---
    char ver[64];
    snprintf(ver, sizeof ver, "%u.%u.%u", VK_API_VERSION_MAJOR(api), VK_API_VERSION_MINOR(api), VK_API_VERSION_PATCH(api));
    j.obj("device")
        .kv("name", g.props.deviceName).kv("api", ver)
        .kv("vendorId", (int64_t)g.props.vendorID).kv("deviceId", (int64_t)g.props.deviceID)
        .kv("driverVersion", (int64_t)g.props.driverVersion)
        .kv("driverName", g.driverName).kv("driverInfo", g.driverInfo)
        .kv("subgroupSize", g.sgSize).kv("subgroupMin", g.sgMin).kv("subgroupMax", g.sgMax)
        .kv("maxSharedMem", g.props.limits.maxComputeSharedMemorySize)
        .kv("maxInvocations", g.props.limits.maxComputeWorkGroupInvocations)
        .kv("timestampPeriodNs", (double)g.props.limits.timestampPeriod)
        .kv("coopStages", stageNames(g.coopStages))
        .kv("hasCoopExt", hasCoop)
        .endObj();
    j.obj("features")
        .kv("cooperativeMatrix", g.f.coopmat).kv("vulkanMemoryModel", g.f.vmm)
        .kv("shaderFloat16", g.f.f16).kv("shaderInt8", g.f.i8).kv("shaderInt16", g.f.i16)
        .kv("storageBuffer16BitAccess", g.f.storage16)
        .kv("subgroupSizeControl", g.f.sgsc).kv("computeFullSubgroups", g.f.fullSubgroups)
        .kv("VK_KHR_shader_bfloat16", g.f.bf16Ext).kv("shaderBFloat16Type", (bool)fb.shaderBFloat16Type)
        .kv("shaderBFloat16CooperativeMatrix", (bool)fb.shaderBFloat16CooperativeMatrix)
        .kv("VK_EXT_shader_float8", g.f.fp8Ext).kv("shaderFloat8", (bool)f8.shaderFloat8)
        .kv("shaderFloat8CooperativeMatrix", (bool)f8.shaderFloat8CooperativeMatrix)
        .endObj();
    j.arr("configs");
    for (const auto &c : coop)
        j.obj().kv("scope", scopeName(c.scope))
            .kv("a", typeName(c.AType)).kv("b", typeName(c.BType))
            .kv("c", typeName(c.CType)).kv("r", typeName(c.ResultType))
            .kv("m", c.MSize).kv("n", c.NSize).kv("k", c.KSize)
            .kv("sat", (bool)c.saturatingAccumulation).endObj();
    j.endArr();

    if (!hasCoop || !g.f.coopmat || !g.f.vmm) {
        fatal = !hasCoop ? "VK_KHR_cooperative_matrix is not exposed by this device"
              : !g.f.coopmat ? "cooperativeMatrix feature is not supported"
              : "vulkanMemoryModel feature is not supported";
        log.line("%s", fatal.c_str());
        j.arr("bench").endArr();
        return finish();
    }

    // --- device ---
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g.pd, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(g.pd, &nq, qf.data());
    int qfi = -1;
    for (uint32_t i = 0; i < nq; ++i)   // prefer a compute-only family, else any with compute
        if ((qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && qf[i].timestampValidBits) { qfi = (int)i; break; }
    for (uint32_t i = 0; i < nq && qfi < 0; ++i)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) qfi = (int)i;
    if (qfi < 0) { fatal = "no compute queue"; j.arr("bench").endArr(); return finish(); }
    g.qfi = (uint32_t)qfi;
    g.tsBits = qf[qfi].timestampValidBits;

    // Enable exactly what was reported (re-using the query structs, already chained).
    fc.cooperativeMatrixRobustBufferAccess = VK_FALSE;
    f2.features = VkPhysicalDeviceFeatures{};
    f2.features.shaderInt16 = g.f.i16;
    fs.uniformAndStorageBuffer16BitAccess = VK_FALSE;
    fs.storagePushConstant16 = VK_FALSE;
    fs.storageInputOutput16 = VK_FALSE;
    fv.vulkanMemoryModelDeviceScope = VK_FALSE;
    fv.vulkanMemoryModelAvailabilityVisibilityChains = VK_FALSE;
    fb.shaderBFloat16DotProduct = VK_FALSE;

    std::vector<const char *> devExts = { VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME };
    for (const char *e : { VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME, VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
                           VK_KHR_16BIT_STORAGE_EXTENSION_NAME, VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME,
                           VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME, EXT_BF16, EXT_FP8 })
        if (hasExt(exts, e)) devExts.push_back(e);

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qci.queueFamilyIndex = g.qfi;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.pNext = &f2;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = (uint32_t)devExts.size();
    dci.ppEnabledExtensionNames = devExts.data();
    r = vkCreateDevice(g.pd, &dci, nullptr, &g.dev);
    if (r != VK_SUCCESS) { g.dev = VK_NULL_HANDLE; fatal = "vkCreateDevice failed (" + std::to_string(r) + ")"; log.line("%s", fatal.c_str()); j.arr("bench").endArr(); return finish(); }
    vkGetDeviceQueue(g.dev, g.qfi, 0, &g.queue);

    // --- shared objects ---
    VkDescriptorSetLayoutBinding binds[3];
    for (uint32_t i = 0; i < 3; ++i)
        binds[i] = { i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    VkDescriptorSetLayoutCreateInfo dslci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dslci.bindingCount = 3;
    dslci.pBindings = binds;
    vkCreateDescriptorSetLayout(g.dev, &dslci, nullptr, &g.dsl);
    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &g.dsl;
    vkCreatePipelineLayout(g.dev, &plci, nullptr, &g.layout);
    VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 };
    VkDescriptorPoolCreateInfo dpci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &ps;
    vkCreateDescriptorPool(g.dev, &dpci, nullptr, &g.pool);
    VkDescriptorSetAllocateInfo dsai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dsai.descriptorPool = g.pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &g.dsl;
    vkAllocateDescriptorSets(g.dev, &dsai, &g.set);
    VkCommandPoolCreateInfo cpci{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = g.qfi;
    vkCreateCommandPool(g.dev, &cpci, nullptr, &g.cmdPool);
    VkCommandBufferAllocateInfo cbai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cbai.commandPool = g.cmdPool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    vkAllocateCommandBuffers(g.dev, &cbai, &g.cmd);
    VkFenceCreateInfo fci{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    vkCreateFence(g.dev, &fci, nullptr, &g.fence);
    if (g.tsBits) {
        VkQueryPoolCreateInfo qpci{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
        qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = kMaxTimed;
        if (vkCreateQueryPool(g.dev, &qpci, nullptr, &g.qpool) != VK_SUCCESS) g.qpool = VK_NULL_HANDLE;
    }
    log.line("Timing: %s", g.qpool ? "GPU timestamps" : "CPU wall clock (no timestamp support)");

    // --- bench list: unique subgroup-scope configurations ---
    for (const auto &c : coop) {
        if (c.scope != VK_SCOPE_SUBGROUP_KHR) continue;
        bool dup = false;
        for (const auto &e : entries) {
            const auto &d = e.cfg;
            if (std::tie(d.AType, d.BType, d.CType, d.ResultType, d.MSize, d.NSize, d.KSize, d.saturatingAccumulation) ==
                std::tie(c.AType, c.BType, c.CType, c.ResultType, c.MSize, c.NSize, c.KSize, c.saturatingAccumulation))
                dup = true;
        }
        if (!dup) { Entry e; e.cfg = c; entries.push_back(e); }
    }

    log.line("");
    log.line("=== %ux%ux%u GEMM, one run per configuration ===", kSize, kSize, kSize);
    for (size_t i = 0; i < entries.size(); ++i) {
        Entry &e = entries[i];
        const auto &c = e.cfg;
        const bool isInt = typeInfo(c.CType) && typeInfo(c.CType)->kind != Kind::Float;
        log.line("[%zu/%zu] %s x %s -> %s  %ux%ux%u%s", i + 1, entries.size(),
                 typeName(c.AType).c_str(), typeName(c.BType).c_str(), typeName(c.ResultType).c_str(),
                 c.MSize, c.NSize, c.KSize, c.saturatingAccumulation ? " sat" : "");
        if (g.lost) { e.status = "skip"; e.reason = "device lost earlier"; }
        else {
            e.reason = unsupportedReason(g, c);
            if (e.reason.empty()) benchEntry(g, am, log, e);
        }
        if (e.status == "ok")
            log.line("      %.3f %s (median %.3f ms, best %.3f ms)  check %s  [%s]",
                     e.value, isInt ? "TOPS" : "TFLOPS", e.ms, e.msBest, e.check.c_str(), e.tileText.c_str());
        else
            log.line("      %s: %s%s%s", e.status.c_str(), e.reason.c_str(),
                     e.check.empty() ? "" : "  ", e.check.c_str());
    }

    j.arr("bench");
    for (const auto &e : entries) {
        const auto &c = e.cfg;
        const bool isInt = typeInfo(c.CType) && typeInfo(c.CType)->kind != Kind::Float;
        j.obj().kv("a", typeName(c.AType)).kv("b", typeName(c.BType))
            .kv("c", typeName(c.CType)).kv("r", typeName(c.ResultType))
            .kv("m", c.MSize).kv("n", c.NSize).kv("k", c.KSize).kv("sat", (bool)c.saturatingAccumulation)
            .kv("status", e.status).kv("reason", e.reason).kv("kernel", e.kernel)
            .kv("unit", isInt ? "TOPS" : "TFLOPS")
            .kv("value", e.value).kv("valueBest", e.valueBest)
            .kv("ms", e.ms).kv("msBest", e.msBest)
            .kv("check", e.check).kv("maxErr", e.maxErr).kv("tile", e.tileText);
        j.arr("tune");
        for (const auto &t : e.tune)
            j.obj().kv("tile", tileDesc(t.t, c.MSize, c.NSize, g.sgSize)).kv("value", t.tops).kv("note", t.note).endObj();
        j.endArr().endObj();
    }
    j.endArr();
    log.line("");
    log.line("done.");
    return finish();
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_minsuk_coopmatprobe_MainActivity_nativeRun(JNIEnv *env, jobject thiz, jobject assetManager) {
    Log log;
    log.env = env;
    log.cb = thiz;
    jclass cls = env->GetObjectClass(thiz);
    log.mid = env->GetMethodID(cls, "onNativeProgress", "(Ljava/lang/String;)V");
    if (!log.mid) env->ExceptionClear();
    AAssetManager *am = AAssetManager_fromJava(env, assetManager);
    std::string json = run(am, log);
    return env->NewStringUTF(json.c_str());
}
