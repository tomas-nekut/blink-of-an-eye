

//const uploadFileEle = document.getElementById("fileInput")
//console.log(uploadFileEle.files[0]);



setTimeout(function () {
    document.querySelectorAll("img").forEach(i => make_the_magic(i));    //"img:not(.zeman)"
}, 1000);

// setInterval(function () {
//     document.querySelectorAll("img").forEach(i => make_the_magic(i));    //"img:not(.zeman)"
// }, 1000);


function make_the_magic(i)
{



    /*let file = fileElement.files[0];
    let formData = new FormData();
    formData.set('file', file);
    axios.post("http://localhost:3001/upload-single-file", formData)
      .then(res => {
      console.log(res)
    })*/


    //if(i.previous_src != i.src){      
    //    i.previous_src = i.src;
        i.src = "localhost:50000/" + encodeURIComponent(i.src);
        //console.log("localhost:50000/" + encodeURIComponent(i.src));
        console.log("a");
    //}
}

function utf8_to_b64( str ) {
    return window.btoa(encodeURIComponent( escape( str )));
}