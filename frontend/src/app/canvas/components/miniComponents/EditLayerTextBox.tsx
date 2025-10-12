
import TextField from "@mui/material/TextField";
// Define the props expected by the LayerForm component
interface Props {
    setNodes: (val: any) => void;
    // shows the selected nodes at a given time
    selectedNodes: any[];
    // allows the update of Label
    updateNodeLabel: (targetID: any, val: any) => void;
    // allows the update of layerType
    updateNodeLayerType: (targetID: any, val: any) => void;
}


// export default function EditLayerTextBox(props){
//     return (
//         <TextField
//             label="Layer label"
//             value={selectedNodes[0].data.label}
//             onChange={(e) => updateNodeLabel(selectedNodes[0], e.target.value)}
//             fullWidth
//             size="small"
//             sx={{ mb: 2 }}
//         />
//     )
// }
