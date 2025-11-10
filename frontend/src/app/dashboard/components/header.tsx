import { Box, Container } from '@mui/material';
import Avatar from '@mui/material/Avatar';
import Button from '@mui/material/Button';
import Link from "next/link"
import AuthenticationButton from '@/app/components/AuthenticationButton';

export default function Header() {
    const possibleRoutes = [
        { title: "Canvas", href: "/canvas", image: "dashboard.svg"},
        { title: "Logout", href: "/auth/logout", image: "logout.svg"},
    ];
    return (
        <Box id="heading">
            <h1 className="title">
                <Link href = "/">Torchweaver</Link>
                <span className="titleSubtext">dashboard</span>
            </h1>


            <div className="toolbar">
                <AuthenticationButton routes={possibleRoutes} />
            </div>
        </Box>
    )
}

