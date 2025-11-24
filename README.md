# TorchWeaver
*A visual, no-code neural network builder that generates executable PyTorch code.*

TorchWeaver is a client-server web application that allows users to **design neural network architectures visually** using a drag-and-drop interface. It lowers the barrier for experimenting with custom architectures by turning graphical designs into clean, runnable PyTorch code.


## Project Overview
Neural networks are powerful but designing architectures often requires both domain knowledge and programming skills. TorchWeaver aims to reduce this gap by making neural network design:

- **Visual** — build models interactively
- **Intuitive** — configure layers without writing code
- **Reliable** — real-time validation and error checking
- **Production-ready** — export clean PyTorch code


## Key Features
### **Frontend (React)**
- Drag-and-drop canvas for building neural networks  
- Editable parameters for each layer  
- Real-time validation and error checking  
- Export neural network architectures as excutable Pytorch code
- User authentication: signup, login, logout
- Dashboard for managing saved architectures  

### **Backend (Flask & SQLite)**
- Database storage of network designs  
- Rest API for saving, retrieving, deleting, updating, exporting architectures  
- PyTorch code generation endpoint  
- SQLite database for persistent storage  


## **Testing**

Where the test files are located:

---

**Frontend:**

- `frontend/src/app/canvas/components/__tests__/`
  - Component tests for canvas (e.g., `ActivatorsForm.test.tsx`, `InputForm.test.tsx`, etc.)

- `frontend/src/app/canvas/hooks/__tests__/`
  - Hook tests (e.g., `useExport.test.ts`, `useSave.test.ts`, `useParse.test.ts`)

- `frontend/src/app/canvas/utils/__tests__/`
  - Utility tests (e.g., `idGenerator.test.ts`, `parameterValidation.test.ts`)

- `frontend/src/app/dashboard/components/__tests__/`
  - Dashboard component tests (e.g., `SearchBar.test.tsx`, `FavouriteButton.test.tsx`, `DeleteButton.test.tsx`, `Sorting.test.tsx`)

---

Note: That you may see debugging statements like `console.log` in the test files. These were intentionally left in to show evidence of debugging during development.

**Backend:**

- `backend/tests/`
  - All backend test files:
    - `test_generator.py`
    - `test_parse.py`
    - `test_storage.py`
    - `__init__.py`

- `backend/testDatabase.py`
  - Standalone test file for database logic

- `backend/test_db_print.py`
  - Standalone test file for printing database contents

---

**Other:**

- `extra_test_files/`
  - (Contains aditional test word documents where they documented bugs and their fixes during development)

---

**How to run tests (examples):**
- Run all frontend tests:
  ```bash
  cd frontend
  npm test
  ```
- Run a specific frontend test file:
  ```bash
  npm test -- src/app/dashboard/components/__tests__/SearchBar.test.tsx
  ```
- Run coverage report for frontend:
  ```bash
  npm test -- --coverage
  ```
- Run all backend tests:
  ```bash
  cd backend
  python -m unittest discover -s tests -p "test_*.py" -v
  ```
- Run a specific backend test file:
  ```bash
  python -m unittest tests.test_storage -v
  ```



## Getting Started
1. Clone the repository
```bash 
git clone https://github.com/unsw-cse-comp99-3900/capstone-project-25t3-3900-f09b-cherry.git
cd capstone-project-25t3-3900-f09b-cherry
```
2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
3. Backend Setup
```bash
cd backend
pip install -r requirements.txt
python api.py
```
4. Open http://localhost:3000 with your browser.
