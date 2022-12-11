// In the wink of an eye
// an extension for Google Chrome that makes images of 
// Czech president Miloš Zeman more realistic
//
// Copyright (C) 2022 Tomáš Nekut
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You received a copy of the GNU General Public License
// along with this program or you can find it at <https://www.gnu.org/licenses/>.

url = "https://<IMAGE-PROCESSING SERVER URL>/animate";  

setTimeout(function () {
    document.querySelectorAll("img").forEach(i => animate(i));    
}, 1000);

function base64Encode (buf) {
    let string = '';
    (new Uint8Array(buf)).forEach(byte => string += String.fromCharCode(byte));
    return btoa(string);
}

async function animate(img) {
    const requestOptions = {
        method: 'POST',
        headers: {'Content-Type': 'application/json', 'Bypass-Tunnel-Reminder': 1},
        body: JSON.stringify({"img_url": img.src})
    };
    fetch(url, requestOptions)
        .then(response => response.status == 200 ? response : null)
        .then(response => response.arrayBuffer())
        .then(buff => base64Encode(buff))
        .then(base64 => img.src = 'data:image/png;base64,' + base64)
        .then(_ => img.srcset = "");
}