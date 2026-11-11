package com.minsuk.coopmatprobe;

import android.app.Activity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.os.Message;
import android.os.Messenger;
import android.os.RemoteException;
import android.util.Log;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.CheckBox;
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


    // Same order as kVariants in coopmat_probe.cpp (bit i = path i).
    private static final String[] PATHS = { "no LDS", "LDS x1", "LDS x2", "LDS x3" };
    private final CheckBox[] pathBoxes = new CheckBox[PATHS.length];

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

        LinearLayout paths = new LinearLayout(this);
        paths.setOrientation(LinearLayout.HORIZONTAL);
        paths.setPadding(0, dp(6), 0, 0);
        for (int i = 0; i < PATHS.length; i++) {
            CheckBox cb = new CheckBox(this);
            cb.setText(PATHS[i]);
            cb.setTextColor(TEXT);
            cb.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12);
            cb.setChecked(true);
            pathBoxes[i] = cb;
            paths.addView(cb, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
        }
        HorizontalScrollView phs = new HorizontalScrollView(this);
        phs.addView(paths);
        root.addView(phs);

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
        showPreviousCrash();
    }

    // -------------------------------------------------- crash breadcrumb file

    private static final int MAX_ATTEMPTS = 20;

    private File traceFile() {
        return new File(BenchService.dir(this), BenchService.TRACE_FILE);
    }

    private List<String> readTrace() {
        List<String> lines = new ArrayList<>();
        try (java.io.BufferedReader r = new java.io.BufferedReader(
                new java.io.InputStreamReader(new java.io.FileInputStream(traceFile()), StandardCharsets.UTF_8))) {
            String s;
            while ((s = r.readLine()) != null) lines.add(s);
        } catch (Exception ignored) { }
        return lines;
    }

    private String appVersion() {
        try { return getPackageManager().getPackageInfo(getPackageName(), 0).versionName; }
        catch (Exception e) { return "?"; }
    }

    /** The whole app died last time (e.g. the phone reset): show where it stopped. */
    private void showPreviousCrash() {
        List<String> all = readTrace();
        if (all.isEmpty() || BenchService.END_MARK.equals(all.get(all.size() - 1))) return;
        List<String> lines = new ArrayList<>();
        for (String s : all) if (!s.startsWith("#ENTRY")) lines.add(s);

        int from = Math.max(0, lines.size() - 25);
        StringBuilder tail = new StringBuilder();
        if (from > 0) tail.append(lines.get(0)).append("\n…\n");
        for (int i = from; i < lines.size(); i++) tail.append(lines.get(i)).append('\n');
        final String text = tail.toString().trim();

        LinearLayout c = card("⚠ 이전 측정이 끝나지 않고 종료됨");
        c.addView(text("앱 전체가 종료되었거나 폰이 재시작된 것 같습니다. 마지막으로 실행 중이던 단계(맨 아래 줄)가 원인일 가능성이 큽니다. "
                + "문제가 된 경로의 체크를 끄고 다시 측정할 수 있습니다.", 12, BAD, false));
        TextView lt = text(text, 10, MUTED, false);
        lt.setTypeface(Typeface.MONOSPACE);
        lt.setTextIsSelectable(true);
        lt.setPadding(0, dp(8), 0, dp(8));
        HorizontalScrollView hs = new HorizontalScrollView(this);
        hs.addView(lt);
        c.addView(hs);
        Button cp = button("이 로그 복사");
        cp.setOnClickListener(v -> {
            ClipboardManager cm = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
            cm.setPrimaryClip(ClipData.newPlainText("coopmat crash log", text));
            Toast.makeText(this, "복사했습니다", Toast.LENGTH_SHORT).show();
        });
        c.addView(cp);
        results.addView(c);
    }

    // ------------------------------------------------------------------ run
    //
    // The benchmark runs in BenchService's own process. If the GPU driver kills
    // that process, onServiceDisconnected fires here; we read progress.log to
    // find the (config, path) that was running, then start a new attempt that
    // skips it and reuses every config already finished.

    private int runMask;
    private int attempt;
    private final List<String> crashes = new ArrayList<>();          // "C\tkey\tvariant\tstep"
    private final Map<String, String> doneEntries = new LinkedHashMap<>();
    private ServiceConnection conn;
    private boolean finished;
    private volatile String curConfig = "";

    private final Messenger inbox = new Messenger(new Handler(Looper.getMainLooper(), m -> {
        lastActivity = android.os.SystemClock.uptimeMillis();
        Bundle d = m.getData();
        String a = d.getString("a", "");
        switch (m.what) {
            case BenchService.MSG_PROGRESS:
                logBuf.append(a).append('\n');
                liveLog.setText(logBuf);
                if (a.startsWith("[")) { curConfig = a.trim(); status.setText("측정 중  " + curConfig); }
                break;
            case BenchService.MSG_TRACE:
                if (!a.startsWith("#") && !a.startsWith("  ->"))
                    status.setText("측정 중  " + curConfig + "\n" + a);
                break;
            case BenchService.MSG_PID:
                try { benchPid = Integer.parseInt(a); } catch (NumberFormatException ignored) { }
                break;
            case BenchService.MSG_ENTRY:
                doneEntries.put(a, d.getString("b", ""));
                break;
            case BenchService.MSG_DONE:
                if (finished) break;   // already handled via onBenchDied (process exit won the race)
                finished = true;
                stopService();
                onDone(readFile(a));
                break;
        }
        return true;
    }));

    private void start() {
        if (running) return;
        int m = 0;
        for (int i = 0; i < pathBoxes.length; i++) if (pathBoxes[i].isChecked()) m |= 1 << i;
        if (m == 0) {
            Toast.makeText(this, "측정할 경로를 하나 이상 선택하세요", Toast.LENGTH_SHORT).show();
            return;
        }
        runMask = m;
        attempt = 0;
        crashes.clear();
        doneEntries.clear();
        running = true;
        for (CheckBox cb : pathBoxes) cb.setEnabled(false);
        runBtn.setEnabled(false);
        copyBtn.setEnabled(false);
        results.removeAllViews();
        logBuf.setLength(0);
        liveLog.setText("");
        liveLog.setVisibility(View.VISIBLE);
        status.setText("실행 중…");
        status.setTextColor(ACCENT_F);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        launchAttempt();
    }

    private void launchAttempt() {
        attempt++;
        finished = false;
        benchPid = 0;
        hangKilled = false;
        lastActivity = android.os.SystemClock.uptimeMillis();
        watchdog.removeCallbacks(watchdogTick);
        watchdog.postDelayed(watchdogTick, 1000);
        final int thisAttempt = attempt;
        conn = new ServiceConnection() {
            @Override
            public void onServiceConnected(ComponentName name, IBinder service) {
                Message msg = Message.obtain(null, BenchService.MSG_START, runMask, thisAttempt);
                msg.replyTo = inbox;
                Bundle b = new Bundle();
                b.putString("resume", resumeSpec());
                b.putString("version", appVersion());
                msg.setData(b);
                try {
                    new Messenger(service).send(msg);
                } catch (RemoteException e) {
                    onBenchDied();
                }
            }

            @Override
            public void onServiceDisconnected(ComponentName name) {
                if (!finished) onBenchDied();
            }
        };
        if (!bindService(new Intent(this, BenchService.class), conn, Context.BIND_AUTO_CREATE)) {
            conn = null;
            onDone("{\"error\":\"측정 서비스를 시작하지 못했습니다\"}");
        }
    }

    private void stopService() {
        if (conn != null) {
            try { unbindService(conn); } catch (Exception ignored) { }
            conn = null;
        }
    }

    private String resumeSpec() {
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, String> e : doneEntries.entrySet())
            sb.append("D\t").append(e.getKey()).append('\t').append(e.getValue()).append('\n');
        for (String c : crashes) sb.append(c).append('\n');
        return sb.toString();
    }

    /** The benchmark process died (or was killed by the watchdog) before finishing.
     *  Find out where, then retry without it. */
    private void onBenchDied() {
        if (finished) return;
        finished = true;
        stopService();
        final boolean wasHang = hangKilled;

        List<String> trace = readTrace();
        if (!trace.isEmpty() && BenchService.END_MARK.equals(trace.get(trace.size() - 1))) {
            // The run had finished; the process exit just beat MSG_DONE here.
            onDone(readFile(new File(BenchService.dir(this), BenchService.RESULT_FILE).getAbsolutePath()));
            return;
        }

        // Everything finished so far (also covers messages lost in the crash), and
        // the last config / path / tiling / step of the attempt that died.
        String cfg = null, var = null, tile = null, step = "", stepDetail = "";
        boolean gpuTimeout = false;
        for (String s : trace) {
            if (s.startsWith("#ENTRY\t")) {
                String[] f = s.split("\t", 3);
                if (f.length == 3) doneEntries.put(f[1], f[2]);
            } else if (s.startsWith("#ATTEMPT")) {
                cfg = null; var = null; tile = null; step = ""; stepDetail = ""; gpuTimeout = false;
            } else if (s.startsWith("  . #CFG ")) {
                cfg = s.substring(9).trim(); var = null; tile = null; step = ""; stepDetail = "";
            } else if (s.startsWith("  . #VAR ")) {
                var = s.substring(9).trim(); tile = null; step = ""; stepDetail = "";
            } else if (s.startsWith("  . #TILE ")) {
                tile = s.substring(10).trim(); step = ""; stepDetail = "";
            } else if (s.startsWith("  . !!")) {
                gpuTimeout = true;                            // native exited on a GPU job timeout
            } else if (s.startsWith("  . ") && !s.startsWith("  .   ->")) {
                String t = s.substring(4).trim();
                if (t.startsWith("step:")) {
                    stepDetail = t;                           // e.g. "step: run (1/32 probe)"
                } else {
                    if (t.contains("final run")) tile = null; // past the tuning sweep
                    step = t;                                 // kernel / path / tiling line
                    stepDetail = "";
                }
            }
        }
        if (cfg != null && doneEntries.containsKey(cfg)) cfg = null;   // died after finishing it
        if (!stepDetail.isEmpty()) step = (step.isEmpty() ? "" : step + "  ") + "[" + stepDetail.substring(5).trim() + "]";

        String what = gpuTimeout ? "GPU 작업이 4초 안에 끝나지 않음 → 측정 프로세스를 종료해 GPU 작업 강제 중단 (매우 느리거나 멈춘 커널"
                : wasHang ? "응답 없음 " + (WATCHDOG_MS / 1000) + "초 → 강제 종료 (GPU/드라이버 hang 추정"
                : "측정 프로세스가 종료됨 (GPU 드라이버 crash 추정";
        String where = cfg == null ? "(벤치마크 시작 단계)"
                : cfg + (var != null ? " / " + var : "") + (tile != null ? " / tiling #" + tile : "")
                  + (step.isEmpty() ? "" : "\n      " + step);
        logBuf.append("\n!!! ").append(what).append(", 시도 ").append(attempt).append(")\n    위치: ").append(where).append('\n');
        liveLog.setText(logBuf);

        // Skip just that tiling the first time; if the same path fails again, skip the whole path.
        String target = var == null ? "*" : var;
        if (var != null && tile != null) {
            boolean pathFailedBefore = false;
            for (String c : crashes) if (c.startsWith("C\t" + cfg + "\t" + var + "|")) pathFailedBefore = true;
            if (!pathFailedBefore) target = var + "|" + tile;
        }
        String rec = cfg == null ? null : "C\t" + cfg + "\t" + target + "\t" + (gpuTimeout ? "GPU timeout: " : wasHang ? "hang: " : "crash: ") + step.replace('\t', ' ');
        boolean repeat = false;
        if (rec != null) for (String c : crashes) if (c.startsWith("C\t" + cfg + "\t" + target + "\t")) repeat = true;
        if (rec == null || repeat || attempt >= MAX_ATTEMPTS) {
            logBuf.append("    더 이상 건너뛸 수 없어 중단합니다.\n");
            onDone("{\"error\":\"측정 프로세스가 반복해서 멈추거나 종료되었습니다. 위치: " + where.replace("\"", "'").replace("\n", " ") + "\"}");
            return;
        }
        crashes.add(rec);
        logBuf.append(target.contains("|") ? "    이 tiling만 건너뛰고 이어서 측정합니다…\n"
                : "    이 경로를 건너뛰고 이어서 측정합니다…\n");
        liveLog.setText(logBuf);
        status.setText("측정 프로세스 재시작 중 (시도 " + (attempt + 1) + ")");
        new Handler(Looper.getMainLooper()).postDelayed(this::launchAttempt, 1500);
    }

    // ---- watchdog: a step that never returns (shader compiler or GPU hang) ----

    private static final long WATCHDOG_MS = 30_000;
    private volatile int benchPid = 0;
    private long lastActivity = 0;
    private boolean hangKilled = false;
    private final Handler watchdog = new Handler(Looper.getMainLooper());
    private final Runnable watchdogTick = new Runnable() {
        @Override
        public void run() {
            if (!running || finished) return;
            long idle = android.os.SystemClock.uptimeMillis() - lastActivity;
            if (benchPid > 0 && idle > WATCHDOG_MS && !hangKilled) {
                hangKilled = true;
                android.os.Process.killProcess(benchPid);   // same app uid, so this is allowed
                // onServiceDisconnected -> onBenchDied takes it from here.
            } else if (idle > WATCHDOG_MS / 2) {
                status.setText("측정 중  " + curConfig + "\n응답 대기 " + (idle / 1000) + "초 / " + (WATCHDOG_MS / 1000) + "초…");
            }
            watchdog.postDelayed(this, 1000);
        }
    };


    private String readFile(String path) {
        if (path == null || path.isEmpty()) return "{\"error\":\"결과 파일 없음\"}";
        try {
            byte[] b = java.nio.file.Files.readAllBytes(new File(path).toPath());
            return new String(b, StandardCharsets.UTF_8);
        } catch (Exception e) {
            return "{\"error\":\"결과 파일 읽기 실패: " + e.getMessage() + "\"}";
        }
    }

    @Override
    protected void onDestroy() {
        stopService();
        super.onDestroy();
    }

    private void onDone(String json) {
        running = false;
        for (CheckBox cb : pathBoxes) cb.setEnabled(true);
        runBtn.setEnabled(true);
        runBtn.setText("다시 측정");
        getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        try {
            JSONObject o = new JSONObject(json);
            if (attempt > 1 || !o.has("log")) o.put("log", logBuf.toString());   // keep every attempt's log
            if (!crashes.isEmpty()) o.put("crashes", crashes.size());
            json = o.toString();
            render(o);
            lastReport = buildReport(o);
            String saved = save(lastReport, json);
            copyBtn.setEnabled(true);
            String crashNote = crashes.isEmpty() ? "" : "  (드라이버 crash " + crashes.size() + "회: 해당 경로는 FAIL로 표시)";
            status.setText(o.has("error") ? "오류: " + o.optString("error") : "완료. 저장됨: " + saved + crashNote);
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
