import Link from "next/link"
import { useUser } from "@auth0/nextjs-auth0"
import { useState } from "react";

export default function AuthenticationButton() {
    const { user, isLoading } = useUser();
    const [isToggled, setIsToggled] = useState(false)

    if (isLoading) {
        return (
        <button disabled>Loading...</button>
        )
    }

    if (user) {
        // TO BE DONE: ADD USER INFORMATION HERE :)
        // <pre>{JSON.stringify(user, null, 2)}</pre>
        // Make Profile Picture look clickable
        const nameGiven = user.given_name || user.nickname || user.name || user.email;
        console.log(nameGiven)
        return (
            // maybe make fade for menu better in future
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
                            <a className="dropBtn" href="/auth/logout">
                                <img src="dashboard.svg" className="dropdownPicture" />
                                <h2 className="dropdownOption">Dashboard</h2>
                            </a>
                            <a className="dropBtn" href="/auth/logout">
                                    <img src="logout.svg" className="dropdownPicture"/>
                                    <h2 className="dropdownOption">Logout</h2>
                            </a>
                        </div>
                    )}
                </div>
            </>
        )
    } else {
        return (
            <a className="toolbarBtn" href="/auth/login">Login</a>
        )
    }
}