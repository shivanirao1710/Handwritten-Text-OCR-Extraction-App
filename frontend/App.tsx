import React, { useState, useRef } from 'react';
import {
  SafeAreaView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
  Alert,
  Platform,
  ActivityIndicator,
  ScrollView,
  Image
} from 'react-native';
import { Camera, useCameraDevice, PhotoFile } from 'react-native-vision-camera';
import ImageResizer from 'react-native-image-resizer';
import RNFS from 'react-native-fs';

// --- Main App Component ---
export default function App() {
  // --- State Management ---
  const [hasPermission, setHasPermission] = useState(false);
  const [screen, setScreen] = useState('register'); // 'register', 'login', 'dashboard', 'camera', 'confirmPhoto', 'review'
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [authToken, setAuthToken] = useState<string | null>(null);
  const [userId, setUserId] = useState<number | null>(null);
  const [tickets, setTickets] = useState<{ id: number, extracted_text: string }[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [capturedPhoto, setCapturedPhoto] = useState<PhotoFile | null>(null);
  const [rotation, setRotation] = useState(0);

  const device = useCameraDevice('back');
  const camera = useRef<Camera>(null);

  // --- Configuration ---
  const API_BASE_URL = 'https://4d46d7f64c71.ngrok-free.app';

  // --- Effects ---
  React.useEffect(() => {
    (async () => {
      const status = await Camera.requestCameraPermission();
      setHasPermission(status === 'granted');
    })();
  }, []);

  // --- API & Event Handlers ---

  const handleRegister = async () => {
    if (!username || !password) {
      Alert.alert('Error', 'Please enter both username and password.');
      return;
    }
    if (password.length > 72) {
      Alert.alert('Error', 'Password cannot be longer than 72 characters.');
      return;
    }
    setIsLoading(true);
    const details = new URLSearchParams();
    details.append('username', username);
    details.append('password', password);
    try {
      const response = await fetch(`${API_BASE_URL}/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: details.toString(),
      });
      if (response.ok) {
        Alert.alert('Success', 'Registration successful! Please log in.');
        setScreen('login');
      } else {
        const errorData = await response.json();
        Alert.alert('Registration Failed', errorData.detail || 'An unknown error occurred.');
      }
    } catch {
      Alert.alert('Error', 'An error occurred during registration.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogin = async () => {
    if (!username || !password) {
      Alert.alert('Error', 'Please enter both username and password.');
      return;
    }
    if (password.length > 72) {
      Alert.alert('Error', 'Password cannot be longer than 72 characters.');
      return;
    }
    setIsLoading(true);
    const details = new URLSearchParams();
    details.append('username', username);
    details.append('password', password);
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
        setScreen('dashboard');
      } else {
        const errorData = await response.json();
        Alert.alert('Login Failed', errorData.detail || 'Incorrect username or password.');
      }
    } catch {
      Alert.alert('Error', 'An error occurred during login.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogout = () => {
    setAuthToken(null);
    setUserId(null);
    setUsername('');
    setPassword('');
    setScreen('login');
  };

  const handleCapturePhoto = async () => {
    if (camera.current) {
      setIsLoading(true);
      try {
        const photo = await camera.current.takePhoto({
          flash: 'off',
          enableShutterSound: false,
        });
        setCapturedPhoto(photo);
        setRotation(0);
        setScreen('confirmPhoto');
      } catch (e) {
        console.error("Failed to take photo", e);
        Alert.alert("Error", "Could not capture photo.");
      } finally {
        setIsLoading(false);
      }
    }
  };
const handleRotate = async () => {
  if (!capturedPhoto) return;

  try {
    // Increment rotation
    const newRotation = (rotation + 90) % 360;
    setRotation(newRotation);

    const imagePath = Platform.OS === 'android' ? `file://${capturedPhoto.path}` : capturedPhoto.path;

    // Rotate image physically
    const rotatedImage = await ImageResizer.createResizedImage(
      imagePath,
      1080,   // width
      1080,   // height
      'JPEG',
      100,    // quality
      newRotation, // rotation
      undefined,
      false,  // don't keep EXIF
      { mode: 'contain', onlyScaleDown: false }
    );

    // Replace capturedPhoto with rotated one
    // Remove 'file://' prefix for Android
    const uri = rotatedImage.uri.startsWith('file://') ? rotatedImage.uri.replace('file://', '') : rotatedImage.uri;

    setCapturedPhoto({ ...capturedPhoto, path: uri });

    // Reset rotation to 0 for Image component, because the image is now physically rotated
    setRotation(0);

  } catch (error) {
    console.error('Rotation Error:', error);
    Alert.alert('Error', 'Could not rotate image.');
  }
};


  const handleConfirmAndScan = async () => {
    if (!capturedPhoto) return;
    setIsLoading(true);
    try {
      const imagePath = Platform.OS === 'android' ? `file://${capturedPhoto.path}` : capturedPhoto.path;
      const formData = new FormData();
      formData.append('file', {
        uri: imagePath,
        type: 'image/jpeg',
        name: 'ticket.jpg',
      } as any);

      const response = await fetch(`${API_BASE_URL}/scan`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${authToken}` },
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        Alert.alert('Scan Successful', `Extracted Text: ${result.extracted_text}`);
        setScreen('dashboard');
      } else {
        const errorData = await response.json();
        Alert.alert('Scan Failed', errorData.detail || 'Could not process the image.');
      }
    } catch (error) {
      console.error('Scan Error:', error);
      Alert.alert('Error', 'An unexpected error occurred while scanning.');
    } finally {
      setIsLoading(false);
      setCapturedPhoto(null);
    }
  };

  const handleReviewTickets = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/tickets`, {
        method: 'GET',
        headers: { 'Authorization': `Bearer ${authToken}` },
      });
      if (response.ok) {
        const data = await response.json();
        setTickets(data);
        setScreen('review');
      } else {
        const errorData = await response.json();
        Alert.alert('Fetch Failed', errorData.detail || 'Could not retrieve tickets.');
      }
    } catch {
      Alert.alert('Error', 'An error occurred while fetching tickets.');
    } finally {
      setIsLoading(false);
    }
  };

  // --- UI Screens ---
  const renderAuthScreen = (type: 'login' | 'register') => (
    <View style={styles.container}>
      <Text style={styles.title}>{type === 'login' ? 'Login' : 'Register'}</Text>
      <TextInput
        style={styles.input}
        placeholder="Username"
        value={username}
        onChangeText={setUsername}
        autoCapitalize="none"
      />
      <TextInput
        style={styles.input}
        placeholder="Password"
        value={password}
        onChangeText={setPassword}
        secureTextEntry
      />
      <TouchableOpacity style={styles.buttonWide} onPress={type === 'login' ? handleLogin : handleRegister} disabled={isLoading}>
        <Text style={styles.buttonText}>{isLoading ? 'Processing...' : (type === 'login' ? 'Login' : 'Register')}</Text>
      </TouchableOpacity>
      <TouchableOpacity onPress={() => setScreen(type === 'login' ? 'register' : 'login')}>
        <Text style={styles.switchText}>
          {type === 'login' ? 'Need an account? Register' : 'Have an account? Login'}
        </Text>
      </TouchableOpacity>
    </View>
  );

  if (screen === 'register') return renderAuthScreen('register');
  if (screen === 'login') return renderAuthScreen('login');

  if (screen === 'dashboard') {
    return (
      <View style={styles.container}>
        <Text style={styles.title}>Dashboard</Text>
        <Text style={styles.subtitle}>Welcome, {username}!</Text>
        <TouchableOpacity style={styles.buttonWide} onPress={() => setScreen('camera')}>
          <Text style={styles.buttonText}>Scan Ticket</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.buttonWide} onPress={handleReviewTickets}>
          <Text style={styles.buttonText}>Review Tickets</Text>
        </TouchableOpacity>
        <TouchableOpacity style={[styles.buttonWide, styles.logoutButton]} onPress={handleLogout}>
          <Text style={styles.buttonText}>Logout</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (screen === 'camera') {
    if (!device || !hasPermission) {
      return <View style={styles.container}><Text>Camera not available or permission denied.</Text></View>;
    }
    return (
      <View style={styles.fullScreen}>
        <Camera
          ref={camera}
          style={StyleSheet.absoluteFill}
          device={device}
          isActive={true}
          photo={true}
        />
        <View style={styles.cameraControls}>
          <TouchableOpacity style={styles.captureButton} onPress={handleCapturePhoto} disabled={isLoading}>
            {isLoading ? <ActivityIndicator color="#fff" /> : <View style={styles.captureInnerButton} />}
          </TouchableOpacity>
        </View>
        <TouchableOpacity style={styles.backButton} onPress={() => setScreen('dashboard')}>
          <Text style={styles.backButtonText}>Back</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (screen === 'confirmPhoto' && capturedPhoto) {
    return (
      <SafeAreaView style={styles.container}>
        <Text style={styles.title}>Confirm Scan</Text>
        <View style={{ alignItems: 'center' }}>
          <Image
            style={[
              styles.previewImage,
              { transform: [{ rotate: `${rotation}deg` }] },
            ]}
            source={{ uri: `file://${capturedPhoto.path}` }}
          />
        </View>
        {isLoading && <ActivityIndicator size="large" style={{ marginVertical: 20 }} />}
        <View style={styles.confirmationControls}>
          <TouchableOpacity style={[styles.button, styles.retakeButton]} onPress={() => setScreen('camera')} disabled={isLoading}>
            <Text style={styles.buttonText}>Retake</Text>
          </TouchableOpacity>
          <TouchableOpacity style={[styles.button, styles.rotateButton]} onPress={handleRotate} disabled={isLoading}>
            <Text style={styles.buttonText}>Rotate</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.button} onPress={handleConfirmAndScan} disabled={isLoading}>
            <Text style={styles.buttonText}>Confirm & Scan</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  if (screen === 'review') {
    return (
      <SafeAreaView style={styles.container}>
        <Text style={styles.title}>Review Tickets</Text>
        <ScrollView>
          {tickets.length > 0 ? (
            tickets.map(ticket => (
              <View key={ticket.id} style={styles.ticket}>
                <Text>{ticket.extracted_text}</Text>
              </View>
            ))
          ) : (
            <Text>No tickets found.</Text>
          )}
        </ScrollView>
        <TouchableOpacity style={styles.buttonWide} onPress={() => setScreen('dashboard')}>
          <Text style={styles.buttonText}>Back to Dashboard</Text>
        </TouchableOpacity>
      </SafeAreaView>
    );
  }

  return <View style={styles.container}><Text>Loading...</Text></View>;
}

// --- Styles ---
const styles = StyleSheet.create({
  container: { flex: 1, padding: 20, backgroundColor: '#f5f5f5', justifyContent: 'center' },
  fullScreen: { flex: 1, backgroundColor: 'black' },
  title: { fontSize: 28, fontWeight: 'bold', marginBottom: 20, textAlign: 'center' },
  subtitle: { fontSize: 18, textAlign: 'center', marginBottom: 30, color: '#666' },
  input: { height: 50, borderColor: '#ccc', borderWidth: 1, borderRadius: 8, marginBottom: 15, paddingHorizontal: 15, backgroundColor: '#fff' },
  button: { backgroundColor: '#007bff', padding: 15, borderRadius: 8, alignItems: 'center', flex: 1, marginHorizontal: 5 },
  buttonWide: { backgroundColor: '#007bff', padding: 15, borderRadius: 8, alignItems: 'center', marginBottom: 10 },
  retakeButton: { backgroundColor: '#6c757d' },
  rotateButton: { backgroundColor: '#17a2b8' },
  logoutButton: { backgroundColor: '#dc3545' },
  buttonText: { color: '#fff', fontSize: 16, fontWeight: 'bold' },
  switchText: { marginTop: 15, textAlign: 'center', color: '#007bff' },
  cameraControls: { position: 'absolute', bottom: 0, width: '100%', padding: 20, flexDirection: 'row', justifyContent: 'center' },
  captureButton: { width: 70, height: 70, borderRadius: 35, backgroundColor: 'rgba(255,255,255,0.5)', justifyContent: 'center', alignItems: 'center' },
  captureInnerButton: { width: 60, height: 60, borderRadius: 30, backgroundColor: 'white' },
  backButton: { position: 'absolute', top: 50, left: 20, padding: 10, backgroundColor: 'rgba(0,0,0,0.5)', borderRadius: 10 },
  backButtonText: { color: 'white', fontSize: 16 },
  ticket: { backgroundColor: '#fff', padding: 15, borderRadius: 8, marginBottom: 10, borderWidth: 1, borderColor: '#eee' },
  previewImage: { width: '100%', height: 300, resizeMode: 'contain', borderRadius: 10, marginBottom: 20 },
  confirmationControls: { flexDirection: 'row', justifyContent: 'space-around' },
});
