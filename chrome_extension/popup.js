let sentimentChart = null;

document.addEventListener("DOMContentLoaded", function () {
  document.getElementById("scrapeBtn").addEventListener("click", function () {
    console.log("Button clicked");
    document.getElementById("loadingContainer").style.display = "flex";
    document.getElementById("noProductFound").style.display = "none";
    document.getElementById("chartContainer").style.display = "block";
    document.getElementById("wordCloudContainer").style.display = "none"; // Hide word cloud container initially
    document.getElementById("productName").style.display = "none"; // Hide product name initially
    console.log("Loading container displayed");

    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
      chrome.scripting.executeScript(
        {
          target: { tabId: tabs[0].id },
          files: ["content.js"],
        },
        function () {
          chrome.tabs.sendMessage(
            tabs[0].id,
            { action: "scrapeUrl" },
            function (response) {
              if (chrome.runtime.lastError) {
                console.error(chrome.runtime.lastError.message);
                document.getElementById("loadingContainer").style.display =
                  "none";
              } else if (response && response.url) {
                console.log("Scraped URL:", response.url);

                fetchWithRetry(
                  "http://localhost:5000/analyze",
                  {
                    method: "POST",
                    headers: {
                      "Content-Type": "application/json",
                    },
                    body: JSON.stringify({ url: response.url }),
                  },
                  5,
                  2000
                )
                  .then((data) => {
                    console.log("Response from Flask server:", data);
                    document.getElementById("loadingContainer").style.display =
                      "none";
                    if (
                      data.product_name !== "Product Name Not Found" &&
                      data.positive_percentage !== 0
                    ) {
                      displayChart(data);
                      document.getElementById("chartContainer").style.display =
                        "block";
                      document.getElementById(
                        "productName"
                      ).textContent = `Product: ${data.product_name}`;
                      document.getElementById("productName").style.display =
                        "block"; // Show product name

                      // Store word cloud image data for display
                      localStorage.setItem("wordCloud", data.word_cloud);
                    } else {
                      document.getElementById("noProductFound").style.display =
                        "block";
                    }
                  })
                  .catch((error) => {
                    document.getElementById("loadingContainer").style.display =
                      "none";
                    document.getElementById("noProductFound").style.display =
                      "block";
                    console.error("Error:", error);
                  });
              } else {
                document.getElementById("loadingContainer").style.display =
                  "none";
                document.getElementById("noProductFound").style.display =
                  "block";
                console.error("Failed to scrape the URL");
              }
            }
          );
        }
      );
    });
  });

  document
    .getElementById("wordCloudBtn")
    .addEventListener("click", function () {
      let wordCloudData = localStorage.getItem("wordCloud");
      if (wordCloudData) {
        displayWordCloud(wordCloudData);
        document.getElementById("wordCloudContainer").style.display = "block"; // Show word cloud container
      }
    });
});

function fetchWithRetry(url, options, retries, delay) {
  return new Promise((resolve, reject) => {
    function fetchRetry(n) {
      fetch(url, options)
        .then((res) => {
          if (!res.ok) throw new Error(res.statusText);
          return res.json();
        })
        .then(resolve)
        .catch((error) => {
          if (n === 1) return reject(error);
          setTimeout(() => fetchRetry(n - 1), delay);
        });
    }
    fetchRetry(retries);
  });
}

function displayChart(data) {
  if (sentimentChart) {
    sentimentChart.destroy(); // Destroy the existing chart instance
  }

  var ctx = document.getElementById("sentimentChart").getContext("2d");
  sentimentChart = new Chart(ctx, {
    type: "pie",
    data: {
      labels: ["Positive", "Neutral", "Negative"],
      datasets: [
        {
          data: [
            data.positive_percentage,
            data.neutral_percentage,
            data.negative_percentage,
          ],
          backgroundColor: ["#36A2EB", "#FFCE56", "#FF6384"],
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          position: "top",
        },
      },
    },
  });
}

function displayWordCloud(wordCloudData) {
  let wordCloudContainer = document.getElementById("wordCloudContainer");
  wordCloudContainer.innerHTML = ""; // Clear previous word cloud

  if (wordCloudData) {
    let img = document.createElement("img");
    img.src = `data:image/png;base64,${wordCloudData}`;
    img.style.width = "100%"; // Adjust width as needed
    img.style.height = "auto"; // Maintain aspect ratio
    wordCloudContainer.appendChild(img);
  }
}
