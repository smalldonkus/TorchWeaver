import Link from "next/link"

export default function Header() {
    return (
        <>
            <div id="heading">
                <h1 className="title">Torchweaver</h1>
                <div className="toolbar">
                    <Link href="/dashboard">
                        <button className="toolbarBtn">Dashboard</button>
                    </Link>
                    <Link href="/login">
                        <button className="toolbarBtn">Login</button>
                    </Link>
                    <Link href="/login">
                        <button className="toolbarBtn">Register</button>
                    </Link>
                </div>
            </div>
        </>
    )
}