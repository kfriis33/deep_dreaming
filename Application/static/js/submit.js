$("#request-image").submit(function(e) {
    e.preventDefault();
    let data = new FormData(this);
    console.log(data)    

    $.post("/request-image", $('#request-image').serialize(), function(response) {
        console.log(response)
    });
});


function indicate_success() {
	alert("We will email you the resulting image shortly!")
}