'use client';

import Link from "next/link"
import AuthenticationButton from "./AuthenticationButton";

export default function Header() {
    return (
        <>
            <div id="heading">
                <h1 className="title">Torchweaver</h1>
                <div className="toolbar">
                    <AuthenticationButton />
                </div>
            </div>
        </>
    )
}