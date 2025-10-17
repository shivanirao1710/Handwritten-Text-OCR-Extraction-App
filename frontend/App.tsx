import React, { useState } from 'react';
import {
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
  Alert,
  ActivityIndicator,
  ScrollView,
  Image,
  Platform,
  Modal,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import DocumentScanner from 'react-native-document-scanner-plugin';

export default function App() {
  // State Management
  const [screen, setScreen] = useState('register');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [authToken, setAuthToken] = useState<string | null>(null);
  const [userId, setUserId] = useState<number | null>(null);
  const [tickets, setTickets] = useState<{ id: number, extracted_text: string, image_url: string }[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [processingImageUri, setProcessingImageUri] = useState<string | null>(null);
  
  // States for edit functionality
  const [editModalVisible, setEditModalVisible] = useState(false);
  const [editingTicket, setEditingTicket] = useState<{ id: number, extracted_text: string, image_url: string } | null>(null);
  const [editedText, setEditedText] = useState('');
  const [isUpdating, setIsUpdating] = useState(false);

  const API_BASE_URL = 'https://3cccff8b71ce.ngrok-free.app';

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

  const handleScanAndUpload = async () => {
    setIsLoading(true);
    try {
      // Open the document scanner
      const { scannedImages, status } = await DocumentScanner.scanDocument();

      // Check if the user cancelled the scan
      if ((status as string) === "cancelled") {
        console.log("Scan was cancelled by the user.");
        setIsLoading(false);
        return;
      }

      // If images are scanned, process the first one
      if (scannedImages && scannedImages.length > 0) {
        const imageUri = scannedImages[0];
        setProcessingImageUri(imageUri); // Set for display on processing screen
        setScreen('processing');

        // Prepare form data for upload
        const formData = new FormData();
        formData.append('file', {
          uri: imageUri,
          type: 'image/jpeg',
          name: 'ticket.jpg',
        });

        // Upload the image to the backend
        const response = await fetch(`${API_BASE_URL}/scan`, {
          method: 'POST',
          headers: { 'Authorization': `Bearer ${authToken}` },
          body: formData,
        });

        if (response.ok) {
          const result = await response.json();
          Alert.alert('Scan Successful', `Extracted Text: ${result.extracted_text}`);
        } else {
          const errorData = await response.json();
          Alert.alert('Scan Failed', errorData.detail || 'Could not process the image.');
        }
      }
    } catch (error) {
      console.error('Scan or Upload Error:', error);
      Alert.alert('Please wait', 'The image is still being processed. This may take a moment.');
    } finally {
      setIsLoading(false);
      setScreen('dashboard'); // Always return to dashboard after processing
      setProcessingImageUri(null); // Clear the processing image
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

  // Function to handle text editing
  const handleEditText = (ticket: { id: number, extracted_text: string, image_url: string }) => {
    setEditingTicket(ticket);
    setEditedText(ticket.extracted_text);
    setEditModalVisible(true);
  };

  // Function to save edited text
  const handleSaveEditedText = async () => {
  if (!editingTicket || !editedText.trim()) {
    Alert.alert('Error', 'Please enter some text');
    return;
  }

  setIsUpdating(true);
  try {
    const response = await fetch(`${API_BASE_URL}/update-ticket-text`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${authToken}`
      },
      body: JSON.stringify({
        ticket_id: editingTicket.id,
        extracted_text: editedText
      }),
    });

    if (response.ok) {
      const result = await response.json();
      
      // Update the local state
      const updatedTickets = tickets.map(ticket =>
        ticket.id === editingTicket.id 
          ? { ...ticket, extracted_text: result.ticket.extracted_text }
          : ticket
      );
      setTickets(updatedTickets);
      
      Alert.alert('Success', 'Text updated successfully!');
      setEditModalVisible(false);
      setEditingTicket(null);
      setEditedText('');
    } else {
      const errorData = await response.json();
      Alert.alert('Update Failed', errorData.detail || 'Failed to update text');
    }
  } catch (error) {
    console.error('Update Error:', error);
    Alert.alert('Error', 'An error occurred while updating the text');
  } finally {
    setIsUpdating(false);
  }
};
  // Function to cancel editing
  const handleCancelEdit = () => {
    setEditModalVisible(false);
    setEditingTicket(null);
    setEditedText('');
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
      <TouchableOpacity 
        style={styles.buttonWide} 
        onPress={type === 'login' ? handleLogin : handleRegister} 
        disabled={isLoading}
      >
        {isLoading ? (
          <ActivityIndicator size="small" color="#fff" />
        ) : (
          <Text style={styles.buttonText}>
            {type === 'login' ? 'Login' : 'Register'}
          </Text>
        )}
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
      <SafeAreaView style={styles.dashboardContainer}>
        <View style={styles.header}>
            <TouchableOpacity style={styles.logoutButton} onPress={handleLogout}>
                <Text style={styles.buttonText}>Logout</Text>
            </TouchableOpacity>
        </View>
        <View style={styles.content}>
            <Text style={styles.title}>Dashboard</Text>
            <Text style={styles.subtitle}>Welcome, {username}!</Text>
            <TouchableOpacity 
              style={styles.buttonWide} 
              onPress={handleScanAndUpload} 
              disabled={isLoading}
            >
              {isLoading ? (
                <ActivityIndicator size="small" color="#fff" />
              ) : (
                <Text style={styles.buttonText}>Scan Ticket</Text>
              )}
            </TouchableOpacity>
            <TouchableOpacity 
              style={styles.buttonWide} 
              onPress={handleReviewTickets}
            >
              <Text style={styles.buttonText}>Review Tickets</Text>
            </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  if (screen === 'processing' && processingImageUri) {
    return (
        <SafeAreaView style={styles.container}>
            <Text style={styles.title}>Processing...</Text>
            <View style={styles.processingContainer}>
                <Image
                    style={[styles.previewImage, styles.processingImage]}
                    source={{ uri: processingImageUri }}
                />
                <View style={styles.overlay}>
                    <ActivityIndicator size="large" color="#fff" />
                    <Text style={styles.processingText}>Uploading and scanning your ticket</Text>
                </View>
            </View>
        </SafeAreaView>
    );
  }

  if (screen === 'review') {
    return (
      <SafeAreaView style={styles.container}>
        <Text style={styles.title}>Review Tickets</Text>
        <ScrollView style={styles.ticketsContainer}>
          {tickets.length > 0 ? (
            tickets.map(ticket => (
              <View key={ticket.id} style={styles.ticket}>
                {ticket.image_url && (
                  <Image
                    source={{ uri: `${API_BASE_URL}${ticket.image_url}` }}
                    style={styles.ticketImage}
                    resizeMode="contain"
                  />
                )}
                <Text style={styles.ticketText}>{ticket.extracted_text}</Text>
                {/* Edit button */}
                <TouchableOpacity 
                  style={styles.editButton}
                  onPress={() => handleEditText(ticket)}
                >
                  <Text style={styles.editButtonText}>Edit Text</Text>
                </TouchableOpacity>
              </View>
            ))
          ) : (
            <Text style={styles.noTicketsText}>No tickets found.</Text>
          )}
        </ScrollView>
        
        <TouchableOpacity style={styles.buttonWide} onPress={() => setScreen('dashboard')}>
          <Text style={styles.buttonText}>Back to Dashboard</Text>
        </TouchableOpacity>

        {/* Edit Modal */}
        <Modal
          animationType="slide"
          transparent={true}
          visible={editModalVisible}
          onRequestClose={handleCancelEdit}
        >
          <View style={styles.modalOverlay}>
            <View style={styles.modalContent}>
              <Text style={styles.modalTitle}>Edit Extracted Text</Text>
              
              <TextInput
                style={styles.textInput}
                multiline
                numberOfLines={8}
                value={editedText}
                onChangeText={setEditedText}
                placeholder="Edit the extracted text here..."
                textAlignVertical="top"
              />
              
              <View style={styles.modalButtons}>
                <TouchableOpacity 
                  style={[styles.modalButton, styles.cancelButton]}
                  onPress={handleCancelEdit}
                  disabled={isUpdating}
                >
                  <Text style={styles.modalButtonText}>Cancel</Text>
                </TouchableOpacity>
                
                <TouchableOpacity 
                  style={[styles.modalButton, styles.saveButton]}
                  onPress={handleSaveEditedText}
                  disabled={isUpdating}
                >
                  {isUpdating ? (
                    <ActivityIndicator size="small" color="#fff" />
                  ) : (
                    <Text style={styles.modalButtonText}>Save</Text>
                  )}
                </TouchableOpacity>
              </View>
            </View>
          </View>
        </Modal>
      </SafeAreaView>
    );
  }

  return (
    <View style={styles.container}>
      <Text>Loading...</Text>
    </View>
  );
}

// Styles
const styles = StyleSheet.create({
  container: { 
    flex: 1, 
    padding: 20, 
    backgroundColor: '#f5f5f5', 
    justifyContent: 'center' 
  },
  dashboardContainer: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    padding: 20,
    alignItems: 'flex-end',
  },
  content: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  logoutButton: {
    backgroundColor: '#dc3545',
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 8,
  },
  title: { 
    fontSize: 28, 
    fontWeight: 'bold', 
    marginBottom: 20, 
    textAlign: 'center' 
  },
  subtitle: { 
    fontSize: 18, 
    textAlign: 'center', 
    marginBottom: 30, 
    color: '#666' 
  },
  input: { 
    height: 50, 
    borderColor: '#ccc', 
    borderWidth: 1, 
    borderRadius: 8, 
    marginBottom: 15, 
    paddingHorizontal: 15, 
    backgroundColor: '#fff' 
  },
  buttonWide: { 
    backgroundColor: '#007bff', 
    padding: 15, 
    borderRadius: 8, 
    alignItems: 'center', 
    marginBottom: 10, 
    alignSelf: 'center', 
    width: '90%' 
  },
  buttonText: { 
    color: '#fff', 
    fontSize: 16, 
    fontWeight: 'bold' 
  },
  switchText: { 
    marginTop: 15, 
    textAlign: 'center', 
    color: '#007bff' 
  },
  ticketsContainer: {
    flex: 1,
    marginBottom: 20,
  },
  ticket: { 
    backgroundColor: '#fff', 
    padding: 15, 
    borderRadius: 8, 
    marginBottom: 10, 
    borderWidth: 1, 
    borderColor: '#eee' 
  },
  ticketImage: {
    width: '100%',
    height: 200,
    borderRadius: 8,
    marginBottom: 10,
  },
  ticketText: {
    fontSize: 16,
    color: '#333',
    marginBottom: 10,
    lineHeight: 20,
  },
  noTicketsText: {
    textAlign: 'center',
    fontSize: 16,
    color: '#666',
    marginTop: 50,
  },
  // Edit button styles
  editButton: {
    backgroundColor: '#28a745',
    padding: 10,
    borderRadius: 6,
    alignItems: 'center',
  },
  editButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: 'bold',
  },
  // Modal styles
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  modalContent: {
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 20,
    width: '100%',
    maxHeight: '80%',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.25,
    shadowRadius: 4,
    elevation: 5,
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 15,
    textAlign: 'center',
  },
  textInput: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    minHeight: 150,
    marginBottom: 20,
    backgroundColor: '#f9f9f9',
  },
  modalButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  modalButton: {
    flex: 1,
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
    marginHorizontal: 5,
  },
  cancelButton: {
    backgroundColor: '#6c757d',
  },
  saveButton: {
    backgroundColor: '#007bff',
  },
  modalButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  previewImage: { 
    width: '100%', 
    height: 300, 
    resizeMode: 'contain', 
    borderRadius: 10, 
    marginBottom: 20 
  },
  processingContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    width: '100%',
  },
  processingImage: {
    opacity: 0.6,
  },
  overlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.4)',
    borderRadius: 10,
  },
  processingText: {
    color: '#fff',
    marginTop: 10,
    fontSize: 16,
    fontWeight: 'bold',
  },
});