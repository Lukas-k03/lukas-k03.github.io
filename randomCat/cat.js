const url = `https://api.thecatapi.com/v1/images/search?limit=1`;
const api_key = "live_sVreRTJmCuckaJwfTzBZhO7D5CyscYHRAcdy4BHYGDVMqsvVILQjBDyJRNSq6NLu"

fetch(url,{headers: {
  'x-api-key': api_key
}})
.then((response) => {
return response.json();
})
.then((data) => {
let imagesData = data;
imagesData.map(function(imageData) {

let image = document.createElement('img');
//use the url from the image object
image.src = `${imageData.url}`;
    
let gridCell = document.createElement('div');
gridCell.classList.add('col');
gridCell.classList.add('col-lg');
gridCell.appendChild(image)
  
document.getElementById('grid').appendChild(gridCell);

});
})
.catch(function(error) {
console.log(error);
});