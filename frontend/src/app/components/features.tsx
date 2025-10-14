import Link from "next/link"
export default function Features() {
    return (
        <>
            <div id="featuresBody">
                <h1 id="featuresTitle">Features</h1>
                <div className="featureColumn">
                    <Component 
                        image={"background-img.jpg"}
                        heading={"Drag-and-Drop Functionality"}
                        notes={"This works for sure!This works for sure!This works for sure!This works for sure!This works for sure!This works for sure!"}
                    />
                    <Component 
                        image={"background-img.jpg"}
                        heading={"Pre-designed Selection of Layers and Tensor Operations"}
                        notes={"This works for sure!This works for sure!This works for sure!This works for sure!This works for sure!This works for sure!"}
                    />
                    <Component 
                        image={"background-img.jpg"}
                        heading={"Export Architecture Capabilities"}
                        notes={"This works for sure!This works for sure!This works for sure!This works for sure!This works for sure!This works for sure!"}
                    />

                    <Component 
                        image={"background-img.jpg"}
                        heading={"Streamlined Access and Design"}
                        notes={"This works for sure!This works for sure!This works for sure!This works for sure!This works for sure!This works for sure!"}
                    />
                </div>
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