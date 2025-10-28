import { Alert, Snackbar } from "@mui/material";

interface Props {
    isOpen: boolean;
    message: string;
}

export default function MySnackBar({isOpen, message}: Props){
    return (
        <Snackbar open={isOpen} anchorOrigin={{vertical: "bottom", horizontal: "right"}}>
            <Alert
                severity="error"
                variant="filled"
                sx={{ width: "100%"}}
            >
                {message}
            </Alert>
        </Snackbar>
    )
}