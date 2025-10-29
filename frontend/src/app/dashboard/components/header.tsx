import { Box, Container } from '@mui/material';
import Avatar from '@mui/material/Avatar';
import Button from '@mui/material/Button';
import Link from "next/link"

export default function Header() {
    return (
        <Box id="heading">
            <h1 className="title">
                <Link href = "/">Torchweaver</Link>
                <span className="titleSubtext">dashboard</span>
            </h1>

            <Box sx={{display: "flex", gap: "1rem", pr: "2rem"}}>
                <Button href="/canvas" sx={{color:"#F1F1F1", p: "1rem 2rem", fontSize: "1.5rem"}}>Create New</Button>
                <Link href="/">
                    <Avatar alt="pfp" src="testpfp.jpg" sx={{ width: 70, height: 70 }}/>
                </Link>
            </Box>
        </Box>
    )
}

