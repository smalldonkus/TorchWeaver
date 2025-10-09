import Link from "next/link"

export default function LandingBody() {
    return (

        <>
        
        <div id="landingBody">
            <div id="landingInfo">
                <h1 id="landingSynopsis">Free, Innovative Neural Network Design</h1>
                <p className="landingDesc">No need for technical expertise, Torchweaver combines with your creative vision to create groundbreaking models.</p>
                <Link href="/canvas">
                    <button id="canvasBtn">Get Started</button>
                </Link>
            </div>
            <div id="landingImgContainer">
                <img id="canvasImg"></img>
            </div>
            
        </div>
        </>
    )
}