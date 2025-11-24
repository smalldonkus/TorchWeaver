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
    
    const dialogTitle = screen.getByText(/Are you sure you want to delete this neural network/i);
    expect(dialogTitle).toBeInTheDocument();
  });

  it('should show Yes and No buttons in dialog', async () => {
    render(<DeleteButton onClick={mockOnClick} />);
    
    const button = screen.getByRole('button');
    await userEvent.click(button);
    
    const yesButton = screen.getByRole('button', { name: /yes/i });
    const noButton = screen.getByRole('button', { name: /no/i });
    
    expect(yesButton).toBeInTheDocument();
    expect(noButton).toBeInTheDocument();
  });

  it('should close dialog when No is clicked', async () => {
    render(<DeleteButton onClick={mockOnClick} />);
    
    const button = screen.getByRole('button');
    await userEvent.click(button);
    
    const noButton = screen.getByRole('button', { name: /no/i });
    await userEvent.click(noButton);
    
    await waitFor(() => {
      expect(screen.queryByText(/Are you sure you want to delete this neural network/i)).not.toBeInTheDocument();
    });
    expect(mockOnClick).not.toHaveBeenCalled();
  });

  it('should call onClick and close dialog when Yes is clicked', async () => {
    render(<DeleteButton onClick={mockOnClick} />);
    
    const button = screen.getByRole('button');
    await userEvent.click(button);
    
    const yesButton = screen.getByRole('button', { name: /yes/i });
    await userEvent.click(yesButton);
    
    await waitFor(() => {
      expect(mockOnClick).toHaveBeenCalledTimes(1);
    });
    
    await waitFor(() => {
      expect(screen.queryByText(/Are you sure you want to delete this neural network/i)).not.toBeInTheDocument();
    });
  });

  it('should not call onClick if dialog is cancelled', async () => {
    render(<DeleteButton onClick={mockOnClick} />);
    
    const button = screen.getByRole('button');
    await userEvent.click(button);
    
    const noButton = screen.getByRole('button', { name: /no/i });
    await userEvent.click(noButton);
    
    expect(mockOnClick).not.toHaveBeenCalled();
  });

  it('should handle click without onClick prop', async () => {
    render(<DeleteButton />);
    
    const button = screen.getByRole('button');
    await userEvent.click(button);
    
    const yesButton = screen.getByRole('button', { name: /yes/i });
    
    // Should not throw error when clicking Yes without onClick
    expect(() => userEvent.click(yesButton)).not.toThrow();
  });
});
