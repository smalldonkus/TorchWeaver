import Link from "next/link"

export default function Header() {
    return (
        <>
            <div id="heading">
                <h1 className="title">Torchweaver</h1>
                <div className="toolbar">
                    <a className="toolbarBtn" href="/auth/login">Login</a>
                    <Link href="/login">
                        <button className="toolbarBtn">Register</button>
                    </Link>
                </div>
            </div>
        </>
    )
}