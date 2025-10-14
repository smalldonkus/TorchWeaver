import Link from "next/link"
export default function Features() {
    return (
        <>
            <div id="featuresBody">
                <h1 id="featuresTitle">Features</h1>
                <div className="featureColumn">
                    <Component 
                        image={"dnd.gif"}
                        heading={"Drag-and-Drop Functionality"}
                        notes={"Easy visual framework that allows a user to construct their desirable neural network architecture with no hard-coding."}
                    />
                    <Component 
                        image={"layers.gif"}
                        heading={"Pre-designed Selection of Layers and Tensor Operations"}
                        notes={"Torchweaver comes with a range of pre-built nodes to place in layers and tensor operations of your choosing. Currently supports the full design of AlexNet."}
                    />
                    <Component 
                        image={"export.png"}
                        heading={"Export Architecture Capabilities"}
                        notes={"When an architecture is visually completed with parameters filled, the code for the framework can be exported and generated with ready-to-go Pytorch, allowing quick use for model training."}
                    />

                    <Component 
                        image={"background-img.jpg"}
                        heading={"Streamlined Access and Design"}
                        notes={"User responsive page design and streamlined access allows you to get creating straight away and with no pay-walls or significant loading times. Create now!"}
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