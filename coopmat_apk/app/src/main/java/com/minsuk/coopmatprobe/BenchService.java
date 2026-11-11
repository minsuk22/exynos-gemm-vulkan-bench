package com.minsuk.coopmatprobe;

import android.app.Service;
import android.content.Intent;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.os.Message;
import android.os.Messenger;
import android.os.RemoteException;

import java.io.File;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

/**
 * Runs the native benchmark in its own process (android:process=":bench").
 *
 * The Vulkan driver can take the whole process down: a shader compiler crash,
 * or an abort after a GPU fault. If that happened in the UI process the app
 * would just vanish. Here only this process dies; MainActivity notices the
 * binder going away, reads progress.log to see what was running, and starts a
 * new attempt that skips the (config, path) that crashed.
 *
 * progress.log is the record that survives a crash. Every line is fsync'ed
 * before the step it describes runs:
 *   plain lines       visible progress
 *   "  . " lines      trace: "#CFG key", "#VAR id", or the GPU step about to run
 *   "#ENTRY\tkey\tjson"  a config that finished (not re-run on the next attempt)
 *   "#ATTEMPT n", "#RUN-START ...", "#RUN-END"
 */
public class BenchService extends Service {
    static { System.loadLibrary("coopmat_probe"); }

    static final int MSG_START = 1, MSG_PROGRESS = 2, MSG_TRACE = 3, MSG_ENTRY = 4, MSG_DONE = 5;
    static final String TRACE_FILE = "progress.log";
    static final String RESULT_FILE = "last_result.json";
    static final String END_MARK = "#RUN-END";

    private native String nativeRun(Object assetManager, int variantMask, String resumeSpec);

    private Messenger client;
    private FileOutputStream trace;
    private volatile boolean busy = false;

    private final Messenger me = new Messenger(new Handler(Looper.getMainLooper(), this::handle));

    @Override
    public IBinder onBind(Intent intent) {
        return me.getBinder();
    }

    static File dir(android.content.Context c) {
        File d = c.getExternalFilesDir(null);
        return d != null ? d : c.getFilesDir();
    }

    private boolean handle(Message m) {
        if (m.what != MSG_START || busy) return true;
        busy = true;
        client = m.replyTo;
        final int mask = m.arg1;
        final int attempt = m.arg2;
        Bundle b = m.getData();
        final String resume = b.getString("resume", "");
        final String version = b.getString("version", "?");

        try {
            trace = new FileOutputStream(new File(dir(this), TRACE_FILE), attempt > 1);
        } catch (Exception e) {
            trace = null;
        }
        if (attempt <= 1)
            traceLine("#RUN-START " + new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(new Date())
                    + "  " + Build.MANUFACTURER + " " + Build.MODEL + "  app " + version + "  paths mask " + mask);
        traceLine("#ATTEMPT " + attempt);

        new Thread(() -> {
            String json;
            try {
                json = nativeRun(getAssets(), mask, resume);
            } catch (Throwable t) {
                json = "{\"error\":\"" + t.toString().replace("\"", "'") + "\"}";
            }
            String path = new File(dir(this), RESULT_FILE).getAbsolutePath();
            try (FileOutputStream f = new FileOutputStream(path)) {
                f.write(json.getBytes(StandardCharsets.UTF_8));
            } catch (Exception e) {
                path = "";
            }
            traceLine(END_MARK);
            try { if (trace != null) trace.close(); } catch (Exception ignored) { }
            trace = null;
            send(MSG_DONE, path, null);
            // Exit so every run starts with fresh driver state (a timed-out GPU job leaks
            // the VkDevice on purpose). The UI treats a death after RUN-END as normal.
            try { Thread.sleep(500); } catch (InterruptedException ignored) { }
            android.os.Process.killProcess(android.os.Process.myPid());
        }, "coopmat-bench").start();
        return true;
    }

    private synchronized void traceLine(String s) {
        if (trace == null) return;
        try {
            trace.write((s + "\n").getBytes(StandardCharsets.UTF_8));
            trace.getFD().sync();
        } catch (Exception ignored) { }
    }

    private void send(int what, String a, String b) {
        Messenger c = client;
        if (c == null) return;
        Message m = Message.obtain(null, what);
        Bundle d = new Bundle();
        d.putString("a", a);
        if (b != null) d.putString("b", b);
        m.setData(d);
        try { c.send(m); } catch (RemoteException ignored) { }
    }

    // ---- called from native code on the benchmark thread ----

    @SuppressWarnings("unused")
    public void onNativeProgress(String line) {
        traceLine(line);
        send(MSG_PROGRESS, line, null);
    }

    @SuppressWarnings("unused")
    public void onNativeTrace(String line) {
        traceLine("  . " + line);
        send(MSG_TRACE, line, null);
    }

    @SuppressWarnings("unused")
    public void onNativeEntry(String key, String json) {
        traceLine("#ENTRY\t" + key + "\t" + json);
        send(MSG_ENTRY, key, json);
    }
}
