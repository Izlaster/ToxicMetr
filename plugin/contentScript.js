const apiKey = 'YOUTUBE_API_KEY';
let positive = 0;
let negative = 0;

chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
  if (request.action === 'extractComments') {
    const videoId = getYouTubeVideoId();
    positive = 0;
    negative = 0;
    if (videoId) {
      fetchComments(videoId);
    }
  }
});

function getYouTubeVideoId() {
  const urlParams = new URLSearchParams(window.location.search);
  return urlParams.get('v');
}

async function getClass(textToClassify) {
  const apiUrl = 'http://127.0.0.1:8000/api/classify/';
  const data = { text: textToClassify };

  try {
    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    const result = await response.json();
    console.log('Classification Result:', result);
    return result;
  } catch (error) {
    console.error('API Request Error:', error);
    throw error;
  }
}

async function fetchComments(videoId, maxResults = 50) {
  const apiUrl = `https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId=${videoId}&maxResults=${maxResults}&key=${apiKey}`;

  try {
    const response = await fetch(apiUrl);
    const data = await response.json();
    const comments = data.items.map(item => item.snippet.topLevelComment.snippet.textDisplay);

    const results = await Promise.all(comments.map(comment => getClass(comment)));

    results.forEach(result => {
      if (result['probabilities'][0] === 0) positive += 1;
      else negative += 1;
    });

    const nextPageToken = data.nextPageToken;
    if (nextPageToken) {
      await fetchNextPage(videoId, nextPageToken);
    }

    // console.log(getResult());
    const dataToSend = getResult();
    chrome.runtime.sendMessage({ data: dataToSend });

  } catch (error) {
    console.error('Error fetching comments:', error);
  }
}

async function fetchNextPage(videoId, pageToken) {
  const apiUrl = `https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId=${videoId}&pageToken=${pageToken}&key=${apiKey}`;

  try {
    const response = await fetch(apiUrl);
    const data = await response.json();
    const comments = data.items.map(item => item.snippet.topLevelComment.snippet.textDisplay);

    const results = await Promise.all(comments.map(comment => getClass(comment)));

    results.forEach(result => {
      if (result['probabilities'][0] === 0) positive += 1;
      else negative += 1;
    });

    const nextPageToken = data.nextPageToken;
    if (nextPageToken) {
      await fetchNextPage(videoId, nextPageToken);
    }
  } catch (error) {
    console.error('Error fetching comments (Next Page):', error);
  }
}

function getResult() {
  return Math.round((negative / (negative + positive)) * 100);
}
