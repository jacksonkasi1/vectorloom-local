import AppKit
import Foundation
import WebKit

final class AppDelegate: NSObject, NSApplicationDelegate, WKNavigationDelegate, WKUIDelegate, WKScriptMessageHandler {
    private var window: NSWindow!
    private var webView: WKWebView!
    private var server: Process?
    private let serverURL = URL(string: "http://127.0.0.1:32145/")!

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        createWindow()
        startServer()
        waitForServer(attempt: 0)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    func applicationWillTerminate(_ notification: Notification) {
        if let server, server.isRunning {
            server.terminate()
            server.waitUntilExit()
        }
    }

    private func createWindow() {
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .default()
        configuration.userContentController.add(self, name: "saveSVG")
        webView = WKWebView(frame: .zero, configuration: configuration)
        webView.navigationDelegate = self
        webView.uiDelegate = self
        webView.setValue(false, forKey: "drawsBackground")
        webView.loadHTMLString("<style>body{font:14px -apple-system;padding:40px;color:#34443a;background:#f2f1e9}</style>Starting VectorLoom…", baseURL: nil)

        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 960, height: 820),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "VectorLoom"
        window.minSize = NSSize(width: 520, height: 640)
        window.center()
        window.contentView = webView
        window.makeKeyAndOrderFront(nil)
    }

    private func startServer() {
        guard let resources = Bundle.main.resourceURL else {
            showStartupError("App resources are missing.")
            return
        }
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("VectorLoom", isDirectory: true)
        let models = support.appendingPathComponent("models", isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: models, withIntermediateDirectories: true)
            let process = Process()
            process.executableURL = resources.appendingPathComponent("vectorloom-local")
            process.currentDirectoryURL = resources
            var environment = ProcessInfo.processInfo.environment
            environment["VECTOR_PORT"] = "32145"
            environment["VECTOR_BIND"] = "127.0.0.1"
            environment["VECTOR_ENABLE_MODEL_ADMIN"] = "1"
            environment.removeValue(forKey: "VECTOR_OFFICIAL_RUNTIME")
            environment.removeValue(forKey: "VECTOR_OFFICIAL_8B_RUNTIME")
            environment["VECTOR_MODEL_DIR"] = models.path
            process.environment = environment
            process.standardOutput = FileHandle.nullDevice
            process.standardError = FileHandle.nullDevice
            try process.run()
            server = process
        } catch {
            showStartupError("Could not start the local engine: \(error.localizedDescription)")
        }
    }

    private func waitForServer(attempt: Int) {
        guard attempt < 80 else {
            showStartupError("The local engine did not start.")
            return
        }
        URLSession.shared.dataTask(with: serverURL.appendingPathComponent("api/health")) { [weak self] _, response, _ in
            DispatchQueue.main.async {
                guard let self else { return }
                if (response as? HTTPURLResponse)?.statusCode == 200 {
                    self.webView.load(URLRequest(url: self.serverURL))
                } else {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
                        self.waitForServer(attempt: attempt + 1)
                    }
                }
            }
        }.resume()
    }

    private func showStartupError(_ message: String) {
        let escaped = message
            .replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
        webView.loadHTMLString("<style>body{font:14px -apple-system;padding:40px;color:#9d2930;background:#f2f1e9}</style>\(escaped)", baseURL: nil)
    }

    func webView(
        _ webView: WKWebView,
        decidePolicyFor navigationResponse: WKNavigationResponse,
        decisionHandler: @escaping (WKNavigationResponsePolicy) -> Void
    ) {
        guard
            let response = navigationResponse.response as? HTTPURLResponse,
            let disposition = response.value(forHTTPHeaderField: "Content-Disposition"),
            disposition.localizedCaseInsensitiveContains("attachment"),
            let url = response.url
        else {
            decisionHandler(.allow)
            return
        }
        decisionHandler(.cancel)
        saveDownload(from: url, suggestedName: navigationResponse.response.suggestedFilename ?? "vectorloom.svg")
    }

    private func saveDownload(from url: URL, suggestedName: String) {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = suggestedName
        panel.allowedFileTypes = ["svg"]
        panel.beginSheetModal(for: window) { [weak self] response in
            guard response == .OK, let destination = panel.url else { return }
            URLSession.shared.dataTask(with: url) { data, _, error in
                DispatchQueue.main.async {
                    if let data {
                        do {
                            try data.write(to: destination, options: .atomic)
                        } catch {
                            self?.showAlert("Could not save SVG", detail: error.localizedDescription)
                        }
                    } else {
                        self?.showAlert("Could not download SVG", detail: error?.localizedDescription ?? "Unknown error")
                    }
                }
            }.resume()
        }
    }

    private func showAlert(_ message: String, detail: String) {
        let alert = NSAlert()
        alert.messageText = message
        alert.informativeText = detail
        alert.alertStyle = .warning
        alert.beginSheetModal(for: window)
    }

    func webView(_ webView: WKWebView, runOpenPanelWith parameters: WKOpenPanelParameters,
                 initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping ([URL]?) -> Void) {
        let panel = NSOpenPanel()
        panel.allowedFileTypes = ["png", "jpg", "jpeg", "webp"]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.beginSheetModal(for: window) { response in
            completionHandler(response == .OK ? panel.urls : nil)
        }
    }

    func userContentController(_ userContentController: WKUserContentController,
                               didReceive message: WKScriptMessage) {
        guard message.name == "saveSVG", message.frameInfo.isMainFrame,
              message.frameInfo.securityOrigin.host == "127.0.0.1",
              message.frameInfo.securityOrigin.port == 32145,
              let svg = message.body as? String else { return }
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "vectorloom.svg"
        panel.allowedFileTypes = ["svg"]
        panel.beginSheetModal(for: window) { [weak self] response in
            guard response == .OK, let destination = panel.url else { return }
            do { try svg.write(to: destination, atomically: true, encoding: .utf8) }
            catch { self?.showAlert("Could not save SVG", detail: error.localizedDescription) }
        }
    }
}

let application = NSApplication.shared
let delegate = AppDelegate()
application.delegate = delegate
application.run()
