chrome.action.onClicked.addListener(function (tab) {
  chrome.scripting.executeScript(
    {
      target: { tabId: tab.id },
      files: ["content.js"],
    },
    function () {
      chrome.tabs.sendMessage(
        tab.id,
        { action: "scrapeUrl" },
        function (response) {
          if (response && response.url) {
            console.log("Scraped URL:", response.url);

            // Send the URL to Flask server
            fetch("http://localhost:5000/analyze", {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({ url: response.url }),
            })
              .then((response) => response.json())
              .then((data) => {
                console.log("Response from Flask server:", data);
              })
              .catch((error) => {
                console.error("Error:", error);
              });
          } else {
            console.error("Failed to scrape the URL");
          }
        }
      );
    }
  );
});
