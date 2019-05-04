function submitImageRequest() {
	$.post("/request-image", "{}", responseJSON => {
		console.log(responseJSON)
	}, "json")
}