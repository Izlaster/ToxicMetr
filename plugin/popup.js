document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('extractComments').addEventListener('click', function() {
      chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
        chrome.tabs.sendMessage(tabs[0].id, { action: 'extractComments' });
      });
    });
  });
  
chrome.runtime.onMessage.addListener(function (message, sender, sendResponse) {
  // Check if the message contains the data you sent from content.js
  if (message.data) {
    // Update the content of the 'message' paragraph in popup.html
    document.getElementById("message").innerText = message.data;
  }
});