document.addEventListener("DOMContentLoaded", ()=>{
  fetchCatImage()
  button()
})

function fetchCatImage(){
  let image = document.getElementById("cat")
  fetch('https://api.thecatapi.com/v1/images/search?api_key=live_sVreRTJmCuckaJwfTzBZhO7D5CyscYHRAcdy4BHYGDVMqsvVILQjBDyJRNSq6NLu')
  .then(resp => resp.json())
  .then(json => image.src = json[0].url) //said to call first arr element . url to get random image
}
function button(){
  let button = document.getElementById("catButton")
  button.addEventListener("click",fetchCatImage)
}
