import Link from "next/link"
import { useUser } from "@auth0/nextjs-auth0"
import { useState } from "react";

interface ProfileItem {
    title: string;
    href: string;
    image: string;
}

interface ProfileProps {
    routes: ProfileItem[];
}

export default function AuthenticationButton({ routes }: ProfileProps) {
    const { user, isLoading } = useUser();
    const [isToggled, setIsToggled] = useState(false)

    if (isLoading) {
        return (
        <button disabled>Loading...</button>
        )
    }

    if (user) {
        const nameGiven = user.given_name || user.nickname || user.name || user.email;
        console.log(nameGiven)
        return (
            <>
                <div className="dropdown" onMouseLeave={() => setIsToggled(false)}>
                    <div className="profile">
                        <button  className="profileBtn" onClick={() => setIsToggled(!isToggled)}>
                            <img src={user.picture} alt="Profile" className="profilePicture"/>
                        </button>
                    </div>

                    {isToggled && (
                        <div className="profileDropdown">
                            <div className="profileInfo">
                                <img src={user.picture} alt="Profile" className="userProfilePicture"/>
                                <div className="userDisplayedInfo">
                                    <h1>{nameGiven}</h1>
                                    <h3>{user.email}</h3>
                                </div>
                            </div>

                            {routes.map((route) => (
                                <a key={route.href} className="dropBtn" href={route.href}>
                                    <img src={route.image} alt={route.title} className="dropdownPicture" />
                                    <h2 className="dropdownOption">{route.title}</h2>
                                </a>
                            ))}
                        </div>
                    )}
                </div>
            </>
        )
    } else {
        return (
            <a className="toolbarBtn" href="/auth/login?returnTo=/dashboard">Login</a>
        )
    }
}