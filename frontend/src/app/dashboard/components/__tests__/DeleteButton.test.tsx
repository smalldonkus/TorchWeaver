import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import userEvent from '@testing-library/user-event';
import DeleteButton from '../DeleteButton';

describe('DeleteButton', () => {
  const mockOnClick = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should render delete icon button', () => {
    render(<DeleteButton onClick={mockOnClick} />);
    
    const deleteIcon = screen.getByTestId('DeleteIcon');
    expect(deleteIcon).toBeInTheDocument();
  });

  it('should open confirmation dialog when clicked', async () => {
    render(<DeleteButton onClick={mockOnClick} />);
    
    const button = screen.getByRole('button');
    await userEvent.click(button);
    
    const dialogTitle = screen.getByText(/Delete Network/i);
    expect(dialogTitle).toBeInTheDocument();
  });

  it('should show Delete and Cancel buttons in dialog', async () => {
    render(<DeleteButton onClick={mockOnClick} />);
    
    const button = screen.getByRole('button');
    await userEvent.click(button);
    
    const deleteButton = screen.getByRole('button', { name: /delete/i });
    const cancelButton = screen.getByRole('button', { name: /cancel/i });
    
    expect(deleteButton).toBeInTheDocument();
    expect(cancelButton).toBeInTheDocument();
  });

  it('should close dialog when Cancel is clicked', async () => {
    render(<DeleteButton onClick={mockOnClick} />);
    
    const button = screen.getByRole('button');
    await userEvent.click(button);
    
    const cancelButton = screen.getByRole('button', { name: /cancel/i });
    await userEvent.click(cancelButton);
    
    await waitFor(() => {
      expect(screen.queryByText(/Delete Network/i)).not.toBeInTheDocument();
    });
    expect(mockOnClick).not.toHaveBeenCalled();
  });

  it('should call onClick and close dialog when Delete is clicked', async () => {
    render(<DeleteButton onClick={mockOnClick} />);
    
    const button = screen.getByRole('button');
    await userEvent.click(button);
    
    const deleteButton = screen.getByRole('button', { name: /delete/i });
    await userEvent.click(deleteButton);
    
    await waitFor(() => {
      expect(mockOnClick).toHaveBeenCalledTimes(1);
    });
    
    await waitFor(() => {
      expect(screen.queryByText(/Delete Network/i)).not.toBeInTheDocument();
    });
  });

  it('should not call onClick if dialog is cancelled', async () => {
    render(<DeleteButton onClick={mockOnClick} />);
    
    const button = screen.getByRole('button');
    await userEvent.click(button);
    
    const cancelButton = screen.getByRole('button', { name: /cancel/i });
    await userEvent.click(cancelButton);
    
    expect(mockOnClick).not.toHaveBeenCalled();
  });

  it('should handle click without onClick prop', async () => {
    render(<DeleteButton />);
    
    const button = screen.getByRole('button');
    await userEvent.click(button);
    
    const deleteButton = screen.getByRole('button', { name: /delete/i });
    
    // Should not throw error when clicking Delete without onClick
    expect(() => userEvent.click(deleteButton)).not.toThrow();
  });
});
