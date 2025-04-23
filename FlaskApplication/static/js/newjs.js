$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').attr('src', e.target.result);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }

    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').hide();
        $('#result-text').text('');
        readURL(this);
    });

    // Predict
   /* $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling API /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                $('.loader').hide();
                $('#result').fadeIn(600);
                displayResult(data);
                console.log('Success!');
            },
        });
    }); */
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);
    
        $(this).hide();
        $('#loader').removeClass('hidden');
    
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                $('#loader').addClass('hidden');
                $('#result').fadeIn(600);
                displayResult(data);
                console.log('Success!');
            },
        });
    });
    
    

    // Color-coded, styled result display
    function displayResult(message) {
        const resultText = document.getElementById("result-text");
        resultText.textContent = '' + message;

        resultText.classList.remove("pneumonia", "normal");

        if (message.toLowerCase().includes("pneumonia")) {
            resultText.classList.add("pneumonia");
        } else if (message.toLowerCase().includes("normal")) {
            resultText.classList.add("normal");
        }
    }
});
