import Link from "next/link"

export default function CanvasPage() {
    return (
        <>
            <h1>Canvas TODO page</h1>

            <div className="routes">
                <Link href="/">
                    <button>Back to landing page</button>
                </Link>
            </div>
        </>
    )
}