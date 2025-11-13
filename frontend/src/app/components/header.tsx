'use client';

import Link from "next/link"
import AuthenticationButton from "./AuthenticationButton";

// Interface set-up for the profile drop-down and its routing to other pages, socuh as a logout button integrated with auth0 and a dashboard.
interface ProfileItem {
    title: string;
    href: string;
    image: string;
}

interface ProfileProps {
    routes: ProfileItem[];
}

// Header route that establishes the toolbar and the routing for specifically the landing page.
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