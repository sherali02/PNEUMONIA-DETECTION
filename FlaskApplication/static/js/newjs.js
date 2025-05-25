$(document).ready(function () {
    $('.image-section').hide();
    $('#result').hide();
    $('#view-heatmap-btn').hide(); // initially hidden

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
        $('#view-heatmap-btn').hide();
        readURL(this);
    });

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
                displayResult(data.result);

                if (data.heatmap) {
                    $('#view-heatmap-btn').fadeIn();
                    $('#view-heatmap-btn').attr('onclick', `window.open('/show_heatmap/${data.heatmap}', '_blank')`);
                } else {
                    $('#view-heatmap-btn').hide();
                }
            },
        });
    });

    function displayResult(message) {
        const resultText = document.getElementById("result-text");
        resultText.innerHTML = ''; // Clear previous
        resultText.classList.remove("pneumonia", "normal");

        if (message.toLowerCase().includes("pneumonia")) {
            const pneumoniaLine = document.createElement('div');
            const severityMatch = message.match(/Severity Score: ([\d.]+)%/);
            const levelMatch = message.match(/Level: (\w+)/);

            pneumoniaLine.textContent = "Pneumonia";
            pneumoniaLine.style.fontFamily = "'Inter', sans-serif";
            pneumoniaLine.style.fontWeight = "700";
            pneumoniaLine.style.color = "#d32f2f"; // bright red
            pneumoniaLine.style.marginBottom = "10px";

            const severityLine = document.createElement('div');
            severityLine.textContent = "Severity Score: " + (severityMatch ? severityMatch[1] : "N/A") + "%";
            severityLine.style.fontFamily = "'Courier New', Courier, monospace";
            severityLine.style.color = "#0d47a1"; 
            severityLine.style.fontWeight = "600";

            const levelLine = document.createElement('div');
            levelLine.textContent = "Level: " + (levelMatch ? levelMatch[1] : "N/A");
            levelLine.style.fontFamily = "'Courier New', Courier, monospace";
            levelLine.style.color = "#0d47a1"; 
            levelLine.style.fontWeight = "600";

            resultText.appendChild(pneumoniaLine);
            resultText.appendChild(severityLine);
            resultText.appendChild(levelLine);

            resultText.classList.add("pneumonia");

        } else if (message.toLowerCase().includes("normal")) {
            resultText.textContent = message;
            resultText.classList.add("normal");
        } else {
            resultText.textContent = message;
        }
    }
});
