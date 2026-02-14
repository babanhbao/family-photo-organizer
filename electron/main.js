const { app, BrowserWindow, dialog } = require("electron");
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");
const http = require("http");
const net = require("net");

const HOST = "127.0.0.1";
const PREFERRED_PORT = 5050;

let mainWindow = null;
let backendProcess = null;
let serverPort = PREFERRED_PORT;
let backendStderrTail = "";
let startupLogPath = "";
let sessionCleared = false;

function appendStartupLog(msg) {
  try {
    if (!startupLogPath) return;
    fs.appendFileSync(startupLogPath, `[${new Date().toISOString()}] ${msg}\n`, "utf8");
  } catch (_) {}
}

function isExecutable(p) {
  try {
    fs.accessSync(p, fs.constants.X_OK);
    return true;
  } catch (_) {
    return false;
  }
}

function isTranslocated() {
  return app.isPackaged && process.execPath.includes("/AppTranslocation/");
}

function ensureDir(p) {
  fs.mkdirSync(p, { recursive: true });
}

function copyDirIfMissing(src, dst) {
  if (!fs.existsSync(src) || fs.existsSync(dst)) {
    return;
  }
  fs.cpSync(src, dst, { recursive: true });
}

function copyFileIfMissing(src, dst) {
  if (!fs.existsSync(src) || fs.existsSync(dst)) {
    return;
  }
  ensureDir(path.dirname(dst));
  fs.copyFileSync(src, dst);
}

function bootstrapAppData(appDataDir) {
  ensureDir(appDataDir);
  if (!app.isPackaged) {
    return;
  }
  const seedDir = path.join(process.resourcesPath, "seed");
  copyDirIfMissing(path.join(seedDir, "db"), path.join(appDataDir, "db"));
  copyDirIfMissing(path.join(seedDir, "trained_faces"), path.join(appDataDir, "trained_faces"));
  copyFileIfMissing(path.join(seedDir, "yolov8n-face.pt"), path.join(appDataDir, "yolov8n-face.pt"));
  copyFileIfMissing(path.join(seedDir, "yolov8n.pt"), path.join(appDataDir, "yolov8n.pt"));
}

function waitForServer(url, timeoutMs = 180000, intervalMs = 500) {
  const started = Date.now();
  return new Promise((resolve, reject) => {
    const tryOnce = () => {
      const req = http.get(url, (res) => {
        let data = "";
        res.on("data", (chunk) => {
          data += chunk.toString();
        });
        res.on("end", () => {
          if (res.statusCode === 200) {
            try {
              const parsed = JSON.parse(data || "{}");
              if (parsed && parsed.ok === true && parsed.service === "family-photo-organizer") {
                resolve();
                return;
              }
            } catch (_) {}
          }
          if (Date.now() - started > timeoutMs) {
            reject(new Error("Backend startup timeout"));
            return;
          }
          setTimeout(tryOnce, intervalMs);
        });
      });
      req.on("error", () => {
        if (Date.now() - started > timeoutMs) {
          reject(new Error("Backend startup timeout"));
          return;
        }
        setTimeout(tryOnce, intervalMs);
      });
      req.setTimeout(3000, () => {
        req.destroy();
      });
    };
    tryOnce();
  });
}

function isPortFree(port, host = HOST) {
  return new Promise((resolve) => {
    const srv = net.createServer();
    srv.once("error", () => resolve(false));
    srv.once("listening", () => {
      srv.close(() => resolve(true));
    });
    srv.listen(port, host);
  });
}

async function pickServerPort(start = PREFERRED_PORT, tries = 40) {
  for (let p = start; p < start + tries; p += 1) {
    const free = await isPortFree(p);
    if (free) {
      return p;
    }
  }
  throw new Error(`No free port found in range ${start}-${start + tries - 1}`);
}

function startBackend(port) {
  const appDataDir = path.join(app.getPath("userData"), "data");
  bootstrapAppData(appDataDir);

  const env = {
    ...process.env,
    APP_HOST: HOST,
    APP_PORT: String(port),
    FPO_APP_DATA_DIR: appDataDir,
    PYTHONUNBUFFERED: "1"
  };

  let command;
  let args;
  let cwd;

  if (app.isPackaged) {
    command = path.join(process.resourcesPath, "backend", "family-photo-server");
    args = [];
    cwd = appDataDir;
  } else {
    const projectRoot = path.join(__dirname, "..");
    const pyInVenv = path.join(projectRoot, "env", "bin", "python");
    command = fs.existsSync(pyInVenv) ? pyInVenv : "python3";
    args = [path.join(projectRoot, "app_yolo.py")];
    cwd = projectRoot;
  }

  appendStartupLog(
    `startBackend command=${command} exists=${fs.existsSync(command)} executable=${isExecutable(command)} cwd=${cwd} appDataDir=${appDataDir} resourcesPath=${process.resourcesPath} port=${port}`
  );

  backendProcess = spawn(command, args, {
    cwd,
    env,
    stdio: ["ignore", "pipe", "pipe"]
  });

  backendStderrTail = "";
  backendProcess.stdout.on("data", (d) => process.stdout.write(`[backend] ${d}`));
  backendProcess.stderr.on("data", (d) => {
    const t = d.toString();
    backendStderrTail = (backendStderrTail + t).slice(-4000);
    process.stderr.write(`[backend] ${d}`);
  });
  backendProcess.on("error", (err) => {
    appendStartupLog(`backend spawn error: ${String(err && err.message ? err.message : err)}`);
    if (!app.isQuitting) {
      dialog.showErrorBox("Backend Launch Failed", String(err && err.message ? err.message : err));
    }
  });
  backendProcess.on("exit", (code, signal) => {
    appendStartupLog(`backend exit code=${code ?? "null"} signal=${signal ?? "null"}`);
    backendProcess = null;
    if (!app.isQuitting) {
      const tail = backendStderrTail.trim();
      dialog.showErrorBox(
        "Backend Stopped",
        `Family Photo backend stopped (code=${code ?? "null"}, signal=${signal ?? "null"}).` +
          (tail ? `\n\nLast error output:\n${tail}` : "")
      );
    }
  });
}

function stopBackend() {
  if (backendProcess && !backendProcess.killed) {
    backendProcess.kill("SIGTERM");
  }
}

function loadingScreenDataUrl(message) {
  const safeMessage = String(message || "Starting...");
  const html = `<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Family Photo Organizer</title>
  <style>
    body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f5f5f7;color:#111;display:flex;align-items:center;justify-content:center;height:100vh}
    .wrap{display:flex;flex-direction:column;align-items:center;gap:14px}
    .spinner{width:44px;height:44px;border:4px solid #d1d5db;border-top-color:#0b6bcb;border-radius:999px;animation:spin 1s linear infinite}
    .text{font-size:14px;color:#444}
    @keyframes spin{to{transform:rotate(360deg)}}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="spinner"></div>
    <div class="text">${safeMessage}</div>
  </div>
</body>
</html>`;
  return `data:text/html;charset=utf-8,${encodeURIComponent(html)}`;
}

function ensureMainWindow() {
  if (mainWindow && !mainWindow.isDestroyed()) {
    return mainWindow;
  }
  mainWindow = new BrowserWindow({
    width: 1360,
    height: 900,
    minWidth: 1100,
    minHeight: 700,
    autoHideMenuBar: true,
    webPreferences: {
      contextIsolation: true,
      sandbox: false
    }
  });

  mainWindow.webContents.on("did-fail-load", (_e, code, desc, validatedURL, isMainFrame) => {
    if (!isMainFrame || code === -3) return;
    appendStartupLog(`did-fail-load code=${code} desc=${desc} url=${validatedURL}`);
  });

  mainWindow.webContents.on("render-process-gone", (_e, details) => {
    appendStartupLog(`renderer gone: ${JSON.stringify(details || {})}`);
    dialog.showErrorBox("Renderer Crashed", JSON.stringify(details || {}, null, 2));
  });

  if (!app.isPackaged) {
    mainWindow.webContents.openDevTools({ mode: "detach" });
  }
  return mainWindow;
}

async function clearSessionIfNeeded(win) {
  if (sessionCleared) return;
  const ses = win.webContents.session;
  try {
    await ses.clearCache();
    await ses.clearStorageData({ storages: ["serviceworkers", "cachestorage"] });
  } catch (_) {}
  sessionCleared = true;
}

async function showLoading(message) {
  const win = ensureMainWindow();
  await clearSessionIfNeeded(win);
  await win.loadURL(loadingScreenDataUrl(message));
}

async function createWindow(url) {
  const win = ensureMainWindow();
  await clearSessionIfNeeded(win);

  let lastErr = null;
  for (let i = 0; i < 25; i += 1) {
    try {
      await win.loadURL(url);
      return;
    } catch (err) {
      lastErr = err;
      appendStartupLog(`loadURL attempt ${i + 1} failed: ${String(err && err.message ? err.message : err)}`);
      await new Promise((r) => setTimeout(r, 500));
    }
  }
  throw lastErr || new Error("Unable to load app URL");
}

app.on("before-quit", () => {
  app.isQuitting = true;
  stopBackend();
});

app.whenReady().then(async () => {
  try {
    startupLogPath = path.join(app.getPath("userData"), "startup.log");
    appendStartupLog("app startup");
    await showLoading("Starting Family Photo backend...");
    if (isTranslocated()) {
      dialog.showErrorBox(
        "Install Required",
        "This app is running from an App Translocation path.\nPlease drag it into /Applications and launch from /Applications."
      );
      appendStartupLog(`translocated execPath=${process.execPath}`);
      app.quit();
      return;
    }
    serverPort = await pickServerPort(PREFERRED_PORT, 50);
    const baseUrl = `http://${HOST}:${serverPort}`;
    startBackend(serverPort);
    await showLoading("Loading AI models. This can take up to 1 minute...");
    await waitForServer(`${baseUrl}/healthz`);
    await showLoading("Opening interface...");
    await createWindow(baseUrl);
  } catch (err) {
    const extra = backendStderrTail ? `\n\nBackend tail:\n${backendStderrTail}` : "";
    appendStartupLog(`startup failed: ${String(err && err.message ? err.message : err)}`);
    dialog.showErrorBox(
      "Startup Failed",
      `${String(err && err.message ? err.message : err)}${extra}\n\nStartup log: ${startupLogPath || "(unavailable)"}`
    );
    app.quit();
  }
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("activate", async () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    try {
      await showLoading("Opening interface...");
      if (!backendProcess) {
        serverPort = await pickServerPort(PREFERRED_PORT, 50);
        startBackend(serverPort);
      }
      const baseUrl = `http://${HOST}:${serverPort}`;
      await waitForServer(`${baseUrl}/healthz`);
      await createWindow(baseUrl);
    } catch (err) {
      appendStartupLog(`activate load failure: ${String(err && err.message ? err.message : err)}`);
      dialog.showErrorBox("Page Load Failed", String(err && err.message ? err.message : err));
    }
  }
});
