package com.minsuk.coopmatprobe;

import android.app.Activity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.HorizontalScrollView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TableLayout;
import android.widget.TableRow;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public class MainActivity extends Activity {
    static { System.loadLibrary("coopmat_probe"); }

    private static final String TAG = "COOPMAT";

    private static final int BG = 0xFF101418;
    private static final int CARD = 0xFF1B2128;
    private static final int LINE = 0xFF2C343D;
    private static final int TEXT = 0xFFE8EDF2;
    private static final int MUTED = 0xFF8E99A4;
    private static final int ACCENT_F = 0xFF4FC3F7;   // float types
    private static final int ACCENT_I = 0xFFFFB74D;   // integer types
    private static final int OK = 0xFF81C784;
    private static final int BAD = 0xFFE57373;

    private native String nativeRun(Object assetManager);

    private LinearLayout results;
    private TextView status;
    private TextView liveLog;
    private Button runBtn, copyBtn;
    private final StringBuilder logBuf = new StringBuilder();
    private String lastReport = "";
    private volatile boolean running = false;

    @Override
    protected void onCreate(Bundle b) {
        super.onCreate(b);
        getWindow().setStatusBarColor(BG);
        getWindow().setNavigationBarColor(BG);

        ScrollView scroll = new ScrollView(this);
        scroll.setBackgroundColor(BG);
        scroll.setFillViewport(true);
        LinearLayout root = vbox();
        root.setPadding(dp(16), dp(20), dp(16), dp(32));
        scroll.addView(root);

        TextView title = text("VK_KHR_cooperative_matrix Probe", 22, TEXT, true);
        root.addView(title);
        TextView sub = text("지원되는 data type별 matrix 크기 조회 + 2048×2048×2048 GEMM 성능 측정", 13, MUTED, false);
        sub.setPadding(0, dp(4), 0, dp(12));
        root.addView(sub);

        LinearLayout btns = new LinearLayout(this);
        btns.setOrientation(LinearLayout.HORIZONTAL);
        runBtn = button("측정 시작");
        runBtn.setOnClickListener(v -> start());
        copyBtn = button("리포트 복사");
        copyBtn.setEnabled(false);
        copyBtn.setOnClickListener(v -> copyReport());
        btns.addView(runBtn, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
        View gap = new View(this);
        btns.addView(gap, new LinearLayout.LayoutParams(dp(10), 1));
        btns.addView(copyBtn, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
        root.addView(btns);

        status = text("버튼을 누르면 장치 정보를 읽고 지원되는 모든 subgroup 설정을 2048³으로 측정합니다. "
                + "설정 수에 따라 수십 초 걸릴 수 있습니다.", 13, MUTED, false);
        status.setPadding(0, dp(12), 0, dp(8));
        root.addView(status);

        results = vbox();
        root.addView(results);

        liveLog = text("", 11, MUTED, false);
        liveLog.setTypeface(Typeface.MONOSPACE);
        liveLog.setTextIsSelectable(true);
        root.addView(liveLog);

        setContentView(scroll);
    }

    // ------------------------------------------------------------------ run

    private void start() {
        if (running) return;
        running = true;
        runBtn.setEnabled(false);
        copyBtn.setEnabled(false);
        results.removeAllViews();
        logBuf.setLength(0);
        liveLog.setText("");
        liveLog.setVisibility(View.VISIBLE);
        status.setText("실행 중…");
        status.setTextColor(ACCENT_F);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        new Thread(() -> {
            String json;
            try {
                json = nativeRun(getAssets());
            } catch (Throwable t) {
                json = "{\"error\":\"" + t.toString().replace("\"", "'") + "\"}";
            }
            final String out = json;
            runOnUiThread(() -> onDone(out));
        }, "coopmat-bench").start();
    }

    /** Called from native code on the benchmark thread. */
    @SuppressWarnings("unused")
    public void onNativeProgress(String line) {
        runOnUiThread(() -> {
            logBuf.append(line).append('\n');
            liveLog.setText(logBuf);
            if (line.startsWith("[")) status.setText("측정 중  " + line.trim());
        });
    }

    private void onDone(String json) {
        running = false;
        runBtn.setEnabled(true);
        runBtn.setText("다시 측정");
        getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        try {
            JSONObject o = new JSONObject(json);
            render(o);
            lastReport = buildReport(o);
            String saved = save(lastReport, json);
            copyBtn.setEnabled(true);
            status.setText(o.has("error") ? "오류: " + o.optString("error") : "완료. 저장됨: " + saved);
            status.setTextColor(o.has("error") ? BAD : OK);
            liveLog.setVisibility(View.GONE);
            Log.i(TAG, "JSON " + json);
        } catch (Exception e) {
            status.setText("결과 해석 실패: " + e);
            status.setTextColor(BAD);
        }
    }

    // --------------------------------------------------------------- render

    private void render(JSONObject o) {
        results.removeAllViews();
        JSONObject dev = o.optJSONObject("device");
        JSONObject feat = o.optJSONObject("features");
        JSONArray configs = o.optJSONArray("configs");
        JSONArray bench = o.optJSONArray("bench");
        int n = o.optInt("problem", 2048);

        if (dev != null) {
            LinearLayout c = card("장치");
            addKv(c, "GPU", dev.optString("name"));
            addKv(c, "Vulkan", dev.optString("api"));
            String drv = (dev.optString("driverName") + " " + dev.optString("driverInfo")).trim();
            if (!drv.isEmpty()) addKv(c, "Driver", drv);
            addKv(c, "Subgroup", dev.optInt("subgroupSize") + "  (range " + dev.optInt("subgroupMin")
                    + "–" + dev.optInt("subgroupMax") + ")");
            addKv(c, "Shared mem", dev.optInt("maxSharedMem") + " B / workgroup");
            addKv(c, "Coopmat stages", dev.optString("coopStages"));
            if (feat != null) {
                List<String> on = new ArrayList<>(), off = new ArrayList<>();
                for (java.util.Iterator<String> it = feat.keys(); it.hasNext(); ) {
                    String k = it.next();
                    (feat.optBoolean(k) ? on : off).add(k);
                }
                addKv(c, "지원", String.join(", ", on));
                if (!off.isEmpty()) addKv(c, "미지원", String.join(", ", off));
            }
            results.addView(c);
        }

        // ---- supported sizes grouped by data type ----
        if (configs != null) {
            LinearLayout c = card("data type별 지원 matrix 크기 (M×N×K)");
            if (configs.length() == 0) {
                c.addView(text("광고된 configuration 없음", 13, MUTED, false));
            } else {
                Map<String, List<String>> groups = new LinkedHashMap<>();
                Map<String, JSONObject> heads = new LinkedHashMap<>();
                for (int i = 0; i < configs.length(); i++) {
                    JSONObject e = configs.optJSONObject(i);
                    String key = e.optString("a") + "|" + e.optString("b") + "|" + e.optString("c") + "|"
                            + e.optString("r") + "|" + e.optString("scope") + "|" + e.optBoolean("sat");
                    if (!groups.containsKey(key)) { groups.put(key, new ArrayList<>()); heads.put(key, e); }
                    groups.get(key).add(e.optInt("m") + "×" + e.optInt("n") + "×" + e.optInt("k"));
                }
                TableLayout t = new TableLayout(this);
                t.setStretchAllColumns(false);
                t.addView(row(true, "A", "B", "C", "Result", "M×N×K", "scope", "sat"));
                for (Map.Entry<String, List<String>> g : groups.entrySet()) {
                    JSONObject e = heads.get(g.getKey());
                    t.addView(row(false, e.optString("a"), e.optString("b"), e.optString("c"), e.optString("r"),
                            String.join("\n", g.getValue()), e.optString("scope"), e.optBoolean("sat") ? "yes" : "-"));
                }
                HorizontalScrollView hs = new HorizontalScrollView(this);
                hs.addView(t);
                c.addView(hs);
                c.addView(note(configs.length() + "개 configuration, " + groups.size() + "개 type 조합"));
            }
            results.addView(c);
        }

        // ---- performance ----
        if (bench != null) {
            LinearLayout c = card(n + "×" + n + "×" + n + " GEMM 성능 (D = A·B)");
            double max = 0;
            for (int i = 0; i < bench.length(); i++) {
                JSONArray vs = bench.optJSONObject(i).optJSONArray("variants");
                for (int k = 0; vs != null && k < vs.length(); k++) {
                    JSONObject v = vs.optJSONObject(k);
                    if ("ok".equals(v.optString("status"))) max = Math.max(max, v.optDouble("value", 0));
                }
            }
            if (bench.length() == 0) c.addView(text("측정 가능한 subgroup configuration 없음", 13, MUTED, false));
            for (int i = 0; i < bench.length(); i++) c.addView(perfRow(bench.optJSONObject(i), max));
            c.addView(note("TFLOPS/TOPS = 2·N³ / median GPU time (timestamp). 설정마다 4가지 메모리 경로를 각각 측정합니다: "
                    + "no LDS = global에서 바로 coopMatLoad, LDS x1 = shared memory 단일 버퍼, "
                    + "LDS x2/x3 = double/triple buffering (다음 slab을 register로 prefetch하며 계산). "
                    + "경로마다 workgroup/warp tiling 12가지를 짧게 돌려 가장 빠른 것으로 10회 측정하고, "
                    + "무작위 64개 출력값을 CPU 결과와 비교(check)합니다. ★ = 가장 빠른 경로. "
                    + "행을 누르면 tiling별 결과가 보입니다."));
            results.addView(c);
        }

        String log = o.optString("log", "");
        if (!log.isEmpty()) {
            LinearLayout c = card("Raw log");
            TextView lt = text(log, 10, MUTED, false);
            lt.setTypeface(Typeface.MONOSPACE);
            lt.setTextIsSelectable(true);
            lt.setVisibility(View.GONE);
            TextView toggle = text("펼치기 ▾", 13, ACCENT_F, false);
            toggle.setOnClickListener(v -> {
                boolean show = lt.getVisibility() != View.VISIBLE;
                lt.setVisibility(show ? View.VISIBLE : View.GONE);
                toggle.setText(show ? "접기 ▴" : "펼치기 ▾");
            });
            c.addView(toggle);
            HorizontalScrollView hs = new HorizontalScrollView(this);
            hs.addView(lt);
            c.addView(hs);
            results.addView(c);
        }
    }

    private View perfRow(JSONObject e, double max) {
        LinearLayout box = vbox();
        box.setPadding(0, dp(10), 0, dp(10));
        boolean ok = "ok".equals(e.optString("status"));
        boolean isInt = "TOPS".equals(e.optString("unit"));
        int accent = isInt ? ACCENT_I : ACCENT_F;
        String unit = e.optString("unit");

        String combo = e.optString("a") + " × " + e.optString("b") + " → " + e.optString("r");
        String shape = e.optInt("m") + "×" + e.optInt("n") + "×" + e.optInt("k") + (e.optBoolean("sat") ? "  sat" : "");

        LinearLayout head = new LinearLayout(this);
        head.setOrientation(LinearLayout.HORIZONTAL);
        head.setGravity(Gravity.CENTER_VERTICAL);
        LinearLayout names = vbox();
        names.addView(text(combo, 15, TEXT, true));
        names.addView(text(shape, 12, MUTED, false));
        head.addView(names, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
        LinearLayout right = vbox();
        right.setGravity(Gravity.END);
        TextView val;
        if (ok) {
            val = text(String.format(Locale.US, "%.2f %s", e.optDouble("value"), unit), 18, accent, true);
        } else {
            val = text("fail".equals(e.optString("status")) ? "FAIL" : "SKIP", 15,
                    "fail".equals(e.optString("status")) ? BAD : MUTED, true);
        }
        val.setGravity(Gravity.END);
        right.addView(val);
        if (ok) {
            TextView bl = text("★ " + e.optString("bestLabel"), 11, MUTED, false);
            bl.setGravity(Gravity.END);
            right.addView(bl);
        }
        head.addView(right);
        box.addView(head);

        JSONArray vars = e.optJSONArray("variants");
        if (vars == null || vars.length() == 0) {
            String why = e.optString("reason");
            box.addView(text(why, 12, "fail".equals(e.optString("status")) ? BAD : MUTED, false));
        } else {
            String bestId = ok ? e.optString("best") : "";
            for (int k = 0; k < vars.length(); k++)
                box.addView(variantRow(vars.optJSONObject(k), max, accent, unit, bestId));
        }

        // Tap: every tiling tried, per variant.
        if (vars != null && vars.length() > 0) {
            TextView tv = text("", 11, MUTED, false);
            tv.setTypeface(Typeface.MONOSPACE);
            StringBuilder sb = new StringBuilder();
            for (int k = 0; k < vars.length(); k++) {
                JSONObject v = vars.optJSONObject(k);
                sb.append("— ").append(v.optString("label")).append("  (").append(v.optString("kernel")).append(")\n");
                if (!v.optString("tile").isEmpty()) sb.append("  chosen: ").append(v.optString("tile")).append('\n');
                JSONArray tune = v.optJSONArray("tune");
                for (int i = 0; tune != null && i < tune.length(); i++) {
                    JSONObject t = tune.optJSONObject(i);
                    String s = t.isNull("value") ? "   -   " : String.format(Locale.US, "%7.3f", t.optDouble("value"));
                    sb.append(s).append("  ").append(t.optString("tile"));
                    if (!t.optString("note").isEmpty()) sb.append("  [").append(t.optString("note")).append("]");
                    sb.append('\n');
                }
                if (tune == null || tune.length() == 0) sb.append("  ").append(v.optString("reason")).append('\n');
                sb.append('\n');
            }
            tv.setText(sb.toString().trim());
            tv.setVisibility(View.GONE);
            tv.setPadding(0, dp(8), 0, 0);
            HorizontalScrollView hs = new HorizontalScrollView(this);
            hs.addView(tv);
            box.addView(hs);
            box.setOnClickListener(v -> tv.setVisibility(tv.getVisibility() == View.VISIBLE ? View.GONE : View.VISIBLE));
        }

        LinearLayout wrap = vbox();
        View sep = new View(this);
        sep.setBackgroundColor(LINE);
        wrap.addView(sep, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 1));
        wrap.addView(box);
        return wrap;
    }

    /** One memory-path line: label | bar | value, then median/best/check underneath. */
    private View variantRow(JSONObject v, double max, int accent, String unit, String bestId) {
        LinearLayout col = vbox();
        col.setPadding(0, dp(8), 0, 0);
        boolean ok = "ok".equals(v.optString("status"));
        boolean best = v.optString("id").equals(bestId);

        LinearLayout line = new LinearLayout(this);
        line.setOrientation(LinearLayout.HORIZONTAL);
        line.setGravity(Gravity.CENTER_VERTICAL);
        TextView lab = text((best ? "★ " : "") + v.optString("label"), 12, best ? TEXT : MUTED, best);
        line.addView(lab, new LinearLayout.LayoutParams(dp(118), ViewGroup.LayoutParams.WRAP_CONTENT));

        LinearLayout track = new LinearLayout(this);
        track.setBackground(round(LINE, dp(3)));
        track.setWeightSum(1f);
        if (ok) {
            View bar = new View(this);
            bar.setBackground(round(best ? accent : (accent & 0x00FFFFFF) | 0x99000000, dp(3)));
            float frac = max > 0 ? (float) (v.optDouble("value") / max) : 0f;
            track.addView(bar, new LinearLayout.LayoutParams(0, dp(6), Math.max(0.01f, Math.min(1f, frac))));
        }
        LinearLayout.LayoutParams tp = new LinearLayout.LayoutParams(0, dp(6), 1);
        tp.rightMargin = dp(8);
        line.addView(track, tp);

        String vs;
        int vc;
        if (ok) { vs = String.format(Locale.US, "%.2f", v.optDouble("value")); vc = best ? accent : TEXT; }
        else if ("fail".equals(v.optString("status"))) { vs = "FAIL"; vc = BAD; }
        else { vs = "SKIP"; vc = MUTED; }
        TextView vt = text(vs, 13, vc, true);
        vt.setGravity(Gravity.END);
        line.addView(vt, new LinearLayout.LayoutParams(dp(56), ViewGroup.LayoutParams.WRAP_CONTENT));
        col.addView(line);

        String detail;
        int dc = MUTED;
        if (ok) {
            String check = v.optString("check");
            detail = String.format(Locale.US, "median %.3f ms · best %.3f ms (%.2f %s) · %s",
                    v.optDouble("ms"), v.optDouble("msBest"), v.optDouble("valueBest"), unit, check);
            if (!check.startsWith("pass")) dc = BAD;
        } else {
            detail = v.optString("reason");
            if (!v.optString("check").isEmpty()) detail += "  (" + v.optString("check") + ")";
            if ("fail".equals(v.optString("status"))) dc = BAD;
        }
        TextView dt = text(detail, 10, dc, false);
        dt.setPadding(dp(118), dp(1), 0, 0);
        col.addView(dt);
        return col;
    }

    // --------------------------------------------------------------- report

    private String buildReport(JSONObject o) {
        StringBuilder sb = new StringBuilder();
        sb.append("CoopMat Probe report  ")
                .append(new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(new Date())).append('\n');
        sb.append("Device model: ").append(Build.MANUFACTURER).append(' ').append(Build.MODEL)
                .append(" (").append(Build.HARDWARE).append("), Android ").append(Build.VERSION.RELEASE).append("\n\n");
        sb.append(o.optString("log", ""));
        JSONArray bench = o.optJSONArray("bench");
        if (bench != null && bench.length() > 0) {
            sb.append("\nSummary (").append(o.optInt("problem"))
                    .append("^3, TFLOPS/TOPS by memory path, * = fastest)\n");
            String hdr = String.format(Locale.US, "%-48s %9s %9s %9s %9s  unit", "", "no LDS", "LDS x1", "LDS x2", "LDS x3");
            sb.append(hdr).append('\n');
            String[] ids = { "direct", "lds1", "lds2", "lds3" };
            for (int i = 0; i < bench.length(); i++) {
                JSONObject e = bench.optJSONObject(i);
                String head = String.format(Locale.US, "%-9s x %-9s -> %-7s %3dx%-3dx%-3d%s", e.optString("a"),
                        e.optString("b"), e.optString("r"), e.optInt("m"), e.optInt("n"), e.optInt("k"),
                        e.optBoolean("sat") ? " sat" : "    ");
                JSONArray vs = e.optJSONArray("variants");
                if (vs == null || vs.length() == 0) {
                    sb.append(head).append("  ").append(e.optString("status")).append(": ")
                            .append(e.optString("reason")).append('\n');
                    continue;
                }
                sb.append(String.format(Locale.US, "%-48s", head));
                String best = "ok".equals(e.optString("status")) ? e.optString("best") : "";
                for (String id : ids) {
                    String cell = "-";
                    for (int k = 0; k < vs.length(); k++) {
                        JSONObject v = vs.optJSONObject(k);
                        if (!id.equals(v.optString("id"))) continue;
                        if ("ok".equals(v.optString("status")))
                            cell = String.format(Locale.US, "%.3f", v.optDouble("value")) + (id.equals(best) ? "*" : " ");
                        else cell = v.optString("status").toUpperCase(Locale.US) + " ";
                    }
                    sb.append(String.format(Locale.US, " %9s", cell));
                }
                sb.append("  ").append(e.optString("unit")).append('\n');
            }
        }
        return sb.toString();
    }

    private String save(String report, String json) {
        try {
            File dir = getExternalFilesDir(null);
            if (dir == null) dir = getFilesDir();
            try (FileOutputStream f = new FileOutputStream(new File(dir, "coopmat_report.txt"))) {
                f.write(report.getBytes(StandardCharsets.UTF_8));
            }
            try (FileOutputStream f = new FileOutputStream(new File(dir, "coopmat_report.json"))) {
                f.write(json.getBytes(StandardCharsets.UTF_8));
            }
            return new File(dir, "coopmat_report.txt").getAbsolutePath();
        } catch (Exception e) {
            return "(저장 실패: " + e.getMessage() + ")";
        }
    }

    private void copyReport() {
        ClipboardManager cm = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
        cm.setPrimaryClip(ClipData.newPlainText("coopmat report", lastReport));
        Toast.makeText(this, "리포트를 클립보드에 복사했습니다", Toast.LENGTH_SHORT).show();
    }

    // -------------------------------------------------------------- widgets

    private int dp(float v) {
        return Math.round(TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v, getResources().getDisplayMetrics()));
    }

    private LinearLayout vbox() {
        LinearLayout l = new LinearLayout(this);
        l.setOrientation(LinearLayout.VERTICAL);
        return l;
    }

    private TextView text(CharSequence s, float sp, int color, boolean bold) {
        TextView t = new TextView(this);
        t.setText(s);
        t.setTextSize(TypedValue.COMPLEX_UNIT_SP, sp);
        t.setTextColor(color);
        if (bold) t.setTypeface(Typeface.DEFAULT_BOLD);
        return t;
    }

    private TextView note(String s) {
        TextView t = text(s, 11, MUTED, false);
        t.setPadding(0, dp(10), 0, 0);
        return t;
    }

    private Button button(String s) {
        Button b = new Button(this);
        b.setText(s);
        b.setAllCaps(false);
        return b;
    }

    private GradientDrawable round(int color, int radius) {
        GradientDrawable d = new GradientDrawable();
        d.setColor(color);
        d.setCornerRadius(radius);
        return d;
    }

    private LinearLayout card(String title) {
        LinearLayout c = vbox();
        c.setBackground(round(CARD, dp(12)));
        c.setPadding(dp(14), dp(12), dp(14), dp(14));
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT);
        lp.topMargin = dp(12);
        c.setLayoutParams(lp);
        TextView t = text(title, 16, TEXT, true);
        t.setPadding(0, 0, 0, dp(8));
        c.addView(t);
        return c;
    }

    private void addKv(LinearLayout parent, String k, String v) {
        LinearLayout r = new LinearLayout(this);
        r.setOrientation(LinearLayout.HORIZONTAL);
        r.setPadding(0, dp(3), 0, dp(3));
        TextView kt = text(k, 13, MUTED, false);
        r.addView(kt, new LinearLayout.LayoutParams(dp(110), ViewGroup.LayoutParams.WRAP_CONTENT));
        r.addView(text(v, 13, TEXT, false), new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
        parent.addView(r);
    }

    private TableRow row(boolean header, String... cells) {
        TableRow r = new TableRow(this);
        for (String s : cells) {
            TextView t = text(s, header ? 12 : 13, header ? MUTED : TEXT, header);
            t.setPadding(0, dp(4), dp(14), dp(4));
            if (!header) t.setTypeface(Typeface.MONOSPACE);
            r.addView(t);
        }
        if (!header) {
            r.setBackground(null);
        }
        return r;
    }
}
