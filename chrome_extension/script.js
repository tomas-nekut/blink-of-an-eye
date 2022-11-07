

url = "https://in-a-blink-of-an-eye.loca.lt/";  

setTimeout(function () {
    document.querySelectorAll("img").forEach(i => animate(i));    
}, 1000);

function base64Encode (buf) {
    console.log(buf);
    let string = '';
    (new Uint8Array(buf)).forEach(
        (byte) => { string += String.fromCharCode(byte) }
      )
    return btoa(string)
}

async function animate(img){
    const requestOptions = {
        method: 'POST',
        headers: { 'Content-Type':'application/json', 'Bypass-Tunnel-Reminder':1 },
        body: JSON.stringify({ "img_url": img.src })
    };
    fetch(url, requestOptions)
        .then(response => response.status == 200 ? response : null)
        .then(response => response.arrayBuffer())
        .then(buff => base64Encode(buff))
        .then(base64 => img.src = 'data:image/png;base64,' + base64);
}