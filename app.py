import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from bank_model import load_model


HOST = "127.0.0.1"
PORT = 8000
MODEL = load_model()


INDEX_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Banking Intent Classifier</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
      tailwind.config = {
        theme: {
          extend: {
            colors: {
              aqua: "#33d6c9",
              ocean: "#0f6f9f",
              deep: "#082f63"
            }
          }
        }
      }
    </script>
  </head>
  <body class="min-h-screen bg-slate-50 text-slate-900">
    <main class="min-h-screen bg-[linear-gradient(135deg,#e8fbff_0%,#f8fbff_45%,#d9efff_100%)]">
      <section class="mx-auto flex min-h-screen w-full max-w-5xl flex-col justify-center px-5 py-8">
        <div class="mb-6 flex items-center justify-between gap-4">
          <div>
            <p class="text-sm font-semibold uppercase tracking-wide text-ocean">Banking77</p>
            <h1 class="mt-1 text-3xl font-bold text-deep sm:text-4xl">Banking Intent Classifier</h1>
          </div>
          <div class="hidden rounded-md bg-white/80 px-4 py-3 text-right shadow-sm ring-1 ring-sky-100 sm:block">
            <p class="text-xs font-medium text-slate-500">Model</p>
            <p class="text-sm font-semibold text-deep">TF-IDF Naive Bayes</p>
          </div>
        </div>

        <div class="grid gap-5 lg:grid-cols-[1.2fr_0.8fr]">
          <section class="rounded-lg bg-white p-5 shadow-sm ring-1 ring-sky-100">
            <label for="query" class="text-sm font-semibold text-deep">Customer query</label>
            <textarea
              id="query"
              rows="7"
              class="mt-2 w-full resize-none rounded-md border border-sky-200 bg-sky-50/50 px-4 py-3 text-base outline-none transition focus:border-aqua focus:bg-white focus:ring-4 focus:ring-cyan-100"
              placeholder="Example: I lost my card and need a replacement"
            ></textarea>

            <div class="mt-4 flex flex-wrap gap-2">
              <button
                id="predictBtn"
                class="rounded-md bg-deep px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-ocean focus:outline-none focus:ring-4 focus:ring-cyan-200"
              >
                Predict intent
              </button>
              <button
                id="sampleBtn"
                class="rounded-md border border-sky-200 bg-white px-4 py-2.5 text-sm font-semibold text-ocean transition hover:bg-sky-50 focus:outline-none focus:ring-4 focus:ring-cyan-100"
              >
                Try sample
              </button>
            </div>

            <p id="status" class="mt-4 min-h-6 text-sm text-slate-500"></p>
          </section>

          <aside class="rounded-lg bg-deep p-5 text-white shadow-sm">
            <p class="text-sm font-medium text-cyan-100">Prediction</p>
            <div id="emptyState" class="mt-8 rounded-md border border-white/15 bg-white/5 p-4 text-sm text-cyan-50">
              Enter a banking support question to see the predicted intent label.
            </div>

            <div id="result" class="hidden">
              <div class="mt-4 rounded-md bg-white p-4 text-slate-900">
                <p class="text-xs font-semibold uppercase tracking-wide text-ocean">Top intent</p>
                <p id="intentName" class="mt-2 break-words text-2xl font-bold text-deep"></p>
                <p id="intentLabel" class="mt-1 text-sm text-slate-500"></p>
                <div class="mt-4 h-2 overflow-hidden rounded-full bg-sky-100">
                  <div id="confidenceBar" class="h-full rounded-full bg-aqua transition-all"></div>
                </div>
                <p id="confidenceText" class="mt-2 text-sm font-medium text-ocean"></p>
              </div>

              <div class="mt-5">
                <p class="text-sm font-semibold text-cyan-50">Top matches</p>
                <div id="topList" class="mt-3 space-y-2"></div>
              </div>
            </div>
          </aside>
        </div>

        <section class="mt-5 grid gap-3 text-sm text-slate-600 sm:grid-cols-3">
          <div class="rounded-md bg-white/80 p-4 ring-1 ring-sky-100">
            <p class="font-semibold text-deep">Train rows</p>
            <p>{{TRAIN_SHAPE}}</p>
          </div>
          <div class="rounded-md bg-white/80 p-4 ring-1 ring-sky-100">
            <p class="font-semibold text-deep">Test rows</p>
            <p>{{TEST_SHAPE}}</p>
          </div>
          <div class="rounded-md bg-white/80 p-4 ring-1 ring-sky-100">
            <p class="font-semibold text-deep">Intent classes</p>
            <p>{{CLASS_COUNT}}</p>
          </div>
        </section>
      </section>
    </main>

    <script>
      const queryEl = document.getElementById("query");
      const statusEl = document.getElementById("status");
      const resultEl = document.getElementById("result");
      const emptyStateEl = document.getElementById("emptyState");
      const intentNameEl = document.getElementById("intentName");
      const intentLabelEl = document.getElementById("intentLabel");
      const confidenceBarEl = document.getElementById("confidenceBar");
      const confidenceTextEl = document.getElementById("confidenceText");
      const topListEl = document.getElementById("topList");

      const samples = [
        "I lost my card and need to order a new one",
        "Why was my cash withdrawal declined?",
        "How can I change my phone number?",
        "I was charged twice for the same transaction"
      ];

      function formatPercent(value) {
        return `${Math.round(value * 100)}%`;
      }

      async function predict() {
        const query = queryEl.value.trim();
        if (!query) {
          statusEl.textContent = "Please enter a banking question first.";
          return;
        }

        statusEl.textContent = "Running classifier...";
        resultEl.classList.add("hidden");
        emptyStateEl.classList.remove("hidden");

        try {
          const response = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query })
          });

          const data = await response.json();
          if (!response.ok) {
            throw new Error(data.error || "Prediction failed");
          }

          const prediction = data.prediction;
          intentNameEl.textContent = prediction.label_text.replaceAll("_", " ");
          intentLabelEl.textContent = `Numeric label: ${prediction.label}`;
          confidenceBarEl.style.width = formatPercent(prediction.confidence);
          confidenceTextEl.textContent = `Relative confidence among top matches: ${formatPercent(prediction.confidence)}`;

          topListEl.innerHTML = "";
          data.top_intents.forEach((item) => {
            const row = document.createElement("div");
            row.className = "rounded-md bg-white/10 px-3 py-2 ring-1 ring-white/10";
            row.innerHTML = `
              <div class="flex items-center justify-between gap-3">
                <span class="break-words text-sm font-medium text-white">${item.label_text.replaceAll("_", " ")}</span>
                <span class="shrink-0 text-xs text-cyan-100">${formatPercent(item.confidence)}</span>
              </div>
            `;
            topListEl.appendChild(row);
          });

          emptyStateEl.classList.add("hidden");
          resultEl.classList.remove("hidden");
          statusEl.textContent = `Processed ${data.tokens.length} unigram/bigram features.`;
        } catch (error) {
          statusEl.textContent = error.message;
        }
      }

      document.getElementById("predictBtn").addEventListener("click", predict);
      document.getElementById("sampleBtn").addEventListener("click", () => {
        queryEl.value = samples[Math.floor(Math.random() * samples.length)];
        predict();
      });
      queryEl.addEventListener("keydown", (event) => {
        if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
          predict();
        }
      });
    </script>
  </body>
</html>"""


class BankingIntentHandler(BaseHTTPRequestHandler):
    def send_json(self, payload, status=200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path != "/":
            self.send_error(404)
            return

        html = (
            INDEX_HTML.replace("{{TRAIN_SHAPE}}", f"{MODEL.train_shape[0]} rows")
            .replace("{{TEST_SHAPE}}", f"{MODEL.test_shape[0]} rows")
            .replace("{{CLASS_COUNT}}", str(len(MODEL.label_names)))
        )
        data = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/api/predict":
            self.send_error(404)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
            query = str(payload.get("query", "")).strip()
        except json.JSONDecodeError:
            self.send_json({"error": "Invalid JSON body."}, status=400)
            return

        if not query:
            self.send_json({"error": "Query is required."}, status=400)
            return

        self.send_json(MODEL.predict(query))

    def log_message(self, format, *args):
        return


def main():
    server = ThreadingHTTPServer((HOST, PORT), BankingIntentHandler)
    print(f"Banking intent app running at http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
