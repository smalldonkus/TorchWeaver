import Link from "next/link"
export default function Features() {
    return (
        <>
            <div id="featuresBody">
                <h1 id="featuresTitle">Features</h1>

                {/* <div className="feature">
                    <div className="featureInfo">
                        <h2 className="featureText">Seamless Integration</h2>
                        <h3 className="featureExplanation">feature information will eventually be placed here.</h3>
                    </div>
                        <div className="featureImg"></div>
                </div>
                <div className="featureSeperator"></div>

                <div className="feature">
                    <div className="featureInfo">
                        <h2 className="featureText">Drag-and-Drop Functionality</h2>
                        <h3 className="featureExplanation">feature information will eventually be placed here.</h3>
                    </div>
                        <div className="featureImg"></div>
                </div>
                <div className="featureSeperator"></div> */}

                <Component 

                    image={"sample.jpg"}
                    heading={"Hello this works!"}
                    notes={"This works for sure!"}
                />








            </div>

            

        </>
    )
}

function Component({ image, heading, notes}) {
    return (
        <div className="featureBody">
            <div className="featureImgContainer">
                <img className="featureImg" src={image} />
            </div>
            <h1 className="featureHeading">{heading}</h1>
            <p className="featureNotes">{notes}</p>
        </div>
    )
}