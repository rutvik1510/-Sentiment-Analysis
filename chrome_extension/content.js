chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "scrapeUrl") {
    sendResponse({ url: window.location.href });
  }
});
