'use client';

import Link from "next/link"
import AuthenticationButton from "./AuthenticationButton";

interface ProfileItem {
    title: string;
    href: string;
    image: string;
}

interface ProfileProps {
    routes: ProfileItem[];
}
export default function Header() {
    const possibleRoutes = [
        { title: "Dashboard", href: "/dashboard", image: "dashboard.svg"},
        { title: "Logout", href: "/auth/logout", image: "logout.svg"},
    ];

    return (
        <>
            <div id="heading">
                <h1 className="title">Torchweaver</h1>
                <div className="toolbar">
                    <AuthenticationButton routes={possibleRoutes} />
                </div>
            </div>
        </>
    )
}