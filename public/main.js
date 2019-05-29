$(function(){
    var socket = io.connect();
    var $enterButton = $('.enter-button');
    var $sentArea = $('.sent-area')
    var $resultArea = $('.result-area');
    var $inputBar = $('.form-control');
    var $aspects = $('.aspects');
    var $aspectTerms = $('.aspect-terms');
    
    // message to validate whether the user has logged on to the webpage from front end
    console.log('* User connected!')
    
    // Listener to detect click on enter button
    $enterButton.click(function(){
        
        var sentence = $inputBar.val();
        // Clear all the items in the sections
        clearText();
        
        // Show the entered sentence on html
        $sentArea.append($('<div>').text('Entered Review: ' + sentence))
        // Pass sentence entered to server
        socket.emit('enterButtonClicked', sentence);
        console.log('Review entered:', sentence)
    });
    
    // Listener to detect keystroke
    $inputBar.keypress(function(event){    
        
        // 13 = "Enter" JavaScript Keycode
        if(event.which == 13){
            event.preventDefault();
            var sentence = $inputBar.val();
            
            // Clear all the items in the sections
            clearText();
            
            // Show the entered sentence on html
            $sentArea.append($('<div>').text('Entered Review: ' + sentence))
            
            // Pass sentence entered to the server
            socket.emit('enterButtonClicked', sentence);
            console.log('Review entered:', sentence)
        }
    });
    
    // Listener to get aspects sent by the server
    socket.on('result4aspects', function(result){
        
        // Show the result on the result area
        console.log("Aspects obtained!")
        $aspects.text(result)
        
    });
    
    // Listener to get terms sent by the server
    socket.on('result4terms', function(result){
        
        // Show the result on the result area
        console.log("Terms obtained!")
        $aspectTerms.text(result)
        
    });
    
    // Clear all the items in the sections
    function clearText(){
        $inputBar.val('');
        $aspects.text('');
        $aspectTerms.text('');
        $sentArea.html('');
    }
})