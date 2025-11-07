import React, {
  createContext,
  useState,
  useContext,
  PropsWithChildren, // 1. ADD THIS IMPORT
} from 'react';
import { Alert } from 'react-native';
import { API_BASE_URL } from '../api/config';

interface AuthContextData {
  authToken: string | null;
  userId: number | null;
  username: string | null;
  isLoading: boolean;
  login: (username: string, pass: string) => Promise<void>;
  register: (username: string, pass: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextData>({} as AuthContextData);

// 2. CHANGE THIS LINE to use PropsWithChildren
export const AuthProvider = ({ children }: PropsWithChildren<{}>) => {
  const [authToken, setAuthToken] = useState<string | null>(null);
  const [userId, setUserId] = useState<number | null>(null);
  const [username, setUsername] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const register = async (user: string, pass: string) => {
    if (!user || !pass) {
      Alert.alert('Error', 'Please enter both username and password.');
      return;
    }
    if (pass.length > 72) {
      Alert.alert('Error', 'Password cannot be longer than 72 characters.');
      return;
    }
    setIsLoading(true);
    const details = new URLSearchParams();
    details.append('username', user);
    details.append('password', pass);
    try {
      const response = await fetch(`${API_BASE_URL}/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: details.toString(),
      });
      if (response.ok) {
        Alert.alert('Success', 'Registration successful! Please log in.');
        // We don't auto-login, just report success
      } else {
        const errorData = await response.json();
        Alert.alert(
          'Registration Failed',
          errorData.detail || 'An unknown error occurred.',
        );
      }
    } catch {
      Alert.alert('Error', 'An error occurred during registration.');
    } finally {
      setIsLoading(false);
    }
  };

  const login = async (user: string, pass: string) => {
    if (!user || !pass) {
      Alert.alert('Error', 'Please enter both username and password.');
      return;
    }
    if (pass.length > 72) {
      Alert.alert('Error', 'Password cannot be longer than 72 characters.');
      return;
    }
    setIsLoading(true);
    const details = new URLSearchParams();
    details.append('username', user);
    details.append('password', pass);
    try {
      const response = await fetch(`${API_BASE_URL}/token`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: details.toString(),
      });
      if (response.ok) {
        const data = await response.json();
        setAuthToken(data.access_token);
        setUserId(data.user_id);
        setUsername(user); // Store the username
      } else {
        const errorData = await response.json();
        Alert.alert(
          'Login Failed',
          errorData.detail || 'Incorrect username or password.',
        );
      }
    } catch {
      Alert.alert('Error', 'An error occurred during login.');
    } finally {
      setIsLoading(false);
    }
  };

  const logout = () => {
    setAuthToken(null);
    setUserId(null);
    setUsername(null);
  };

  return (
    <AuthContext.Provider
      value={{
        authToken,
        userId,
        username,
        isLoading,
        login,
        register,
        logout,
      }}>
      {children}
    </AuthContext.Provider>
  );
};

// Custom hook to easily use the context
export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}