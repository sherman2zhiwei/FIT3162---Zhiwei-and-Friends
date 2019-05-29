//import library
var fs = require('fs')
         , http = require('http')
         , socketio = require('socket.io')
         , express = require('express')
         , path = require('path')
         , app = express();

// library to interact with python script
var ps = require('python-shell');

//set directory path
app.use(express.static(path.join(__dirname, 'public')));

//set up server with port 8080
var server=http.createServer(app).listen(8080, function() {
            console.log('* Web Server is successfully set up!')
            console.log('Listening at: http://localhost:8080');
 });

socketio.listen(server).on('connection', function (socket) {
    
    console.log('* User connected!')
    // Listener to receive sentence passed by the client
    socket.on('enterButtonClicked', function(sentence){
        
        console.log('Review received:', sentence)
        
        // url for aspect aggregation
        req_url_asp_agg = "http://127.0.0.1:5000/predict?sentence="+ sentence
        
        // url for aspect term extraction
        req_url_asp_ext = "http://127.0.0.1:5001/predict?sentence="+ sentence
        
        // declare module input for aspect aggregation and term extraction
        //example: python <input>
        var options_asp_agg = {
                mode: 'text',
                scriptPath: './',
                args: [req_url_asp_agg]
            };
        
        var options_asp_ext = {
                mode: 'text',
                scriptPath: './',
                args: [req_url_asp_ext]
            };
        
        console.log("Sending URL to Aspect Aggregation API:", req_url_asp_agg)
        console.log("Sending URL to Aspect Term Extraction API:", req_url_asp_ext)
        
        // Run python script
        ps.PythonShell.run('asp_get_json.py', options_asp_agg, function (err, results) {
            if (err) {
              throw err;
            };
            // Results is an array consisting of messages collected during execution
            console.log('results: %j', results);
        
            // Send the result obtained to the client
            socket.emit('result4aspects', results);
        });
        
        ps.PythonShell.run('asp_get_json.py', options_asp_ext, function (err, results) {
            if (err) {
              throw err;
            };
            // Results is an array consisting of messages collected during execution
            console.log('results: %j', results);
        
            // Send the result obtained to the client
            socket.emit('result4terms', results);
        });
    });
});