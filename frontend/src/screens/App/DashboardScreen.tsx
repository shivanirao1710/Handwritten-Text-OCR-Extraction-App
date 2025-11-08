import React, { useState } from 'react';
import {
  Text,
  TouchableOpacity,
  View,
  Alert,
  ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import DocumentScanner from 'react-native-document-scanner-plugin';
import { useAuth } from '../../context/AuthContext';
import { styles } from '../../styles/styles';
import { API_BASE_URL } from '../../api/config';
import { Ticket } from '../../types';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { AppStackParamList } from '../../navigation/AppNavigator';

type Props = NativeStackScreenProps<AppStackParamList, 'Dashboard'>;

export default function DashboardScreen({ navigation }: Props) {
  const { logout, username, authToken } = useAuth();
  // We split 'isLoading' into two states for better UI control
  const [isScanning, setIsScanning] = useState(false);
  const [isFetching, setIsFetching] = useState(false);

  const handleScanAndUpload = async () => {
    setIsScanning(true);
    try {
      const { scannedImages, status } = await DocumentScanner.scanDocument();

      if ((status as string) === 'cancelled') {
        console.log('Scan was cancelled by the user.');
        setIsScanning(false);
        return;
      }

      if (scannedImages && scannedImages.length > 0) {
        navigation.navigate('Processing'); // Show processing screen

        // --- NEW LOGIC: Build ONE FormData ---
        const formData = new FormData();
        
        for (const [index, imageUri] of scannedImages.entries()) {
          console.log(`Appending image ${index + 1} to form data...`);
          // Note the key is 'files' (plural) to match the backend
          formData.append('files', {
            uri: imageUri,
            type: 'image/jpeg',
            name: `ticket_page_${index + 1}.jpg`,
          });
        }
        
        // --- NEW LOGIC: Make ONE API call for the entire batch ---
        try {
          const response = await fetch(`${API_BASE_URL}/scan`, {
            method: 'POST',
            headers: { Authorization: `Bearer ${authToken}` },
            body: formData,
            // 'Content-Type': 'multipart/form-data' is set automatically
          });

          if (response.ok) {
            // Batch success!
            Alert.alert(
              'Processing Complete',
              `Successfully processed ${scannedImages.length} image(s).`,
            );
            
            // Re-fetch the full list to include the new ticket
            try {
              const listResponse = await fetch(`${API_BASE_URL}/tickets`, {
                method: 'GET',
                headers: { Authorization: `Bearer ${authToken}` },
              });
              if (listResponse.ok) {
                const data: Ticket[] = await listResponse.json();
                navigation.navigate('Review', { tickets: data }); // Navigate with fresh data
              } else {
                Alert.alert('Fetch Failed', 'Could not retrieve new tickets.');
                navigation.navigate('Dashboard');
              }
            } catch (fetchErr) {
              Alert.alert('Error', 'An error occurred while fetching new tickets.');
              navigation.navigate('Dashboard');
            }

          } else {
            // Batch Scan/Upload failed
            const errorData = await response.json();
            console.error('Failed to process document:', errorData.detail);
            Alert.alert(
              'Processing Failed',
              `Could not process the document. ${errorData.detail || ''}`,
            );
            navigation.navigate('Dashboard'); // Go back
          }
        
        } catch (uploadError) {
          console.error('Error uploading document:', uploadError);
          Alert.alert('Upload Error', 'An error occurred while uploading the document.');
          navigation.navigate('Dashboard');
        }

      } else {
        setIsScanning(false); // No images scanned
      }
    } catch (error) {
      console.error('Scan Error:', error);
      Alert.alert(
        'Error',
        'An error occurred during the scan. Please try again.',
      );
      navigation.navigate('Dashboard');
    }
    // Note: We don't set isScanning(false) here because navigation is handling the screen change.
  };

  const handleReviewTickets = async () => {
    setIsFetching(true);
    try {
      const response = await fetch(`${API_BASE_URL}/tickets`, {
        method: 'GET',
        headers: { Authorization: `Bearer ${authToken}` },
      });
      if (response.ok) {
        const data: Ticket[] = await response.json();
        // Pass the fetched tickets as a param to the Review screen
        navigation.navigate('Review', { tickets: data });
      } else {
        const errorData = await response.json();
        Alert.alert(
          'Fetch Failed',
          errorData.detail || 'Could not retrieve tickets.',
        );
      }
    } catch {
      Alert.alert('Error', 'An error occurred while fetching tickets.');
    } finally {
      setIsFetching(false);
    }
  };

  const isLoading = isScanning || isFetching;

  return (
    <SafeAreaView style={styles.dashboardContainer}>
      <View style={styles.header}>
        <TouchableOpacity style={styles.logoutButton} onPress={logout}>
          <Text style={styles.buttonText}>Logout</Text>
        </TouchableOpacity>
      </View>
      <View style={styles.content}>
        <Text style={styles.title}>Dashboard</Text>
        <Text style={styles.subtitle}>Welcome, {username}!</Text>
        <TouchableOpacity
          style={styles.buttonWide}
          onPress={handleScanAndUpload}
          disabled={isLoading}>
          {isScanning ? (
            <ActivityIndicator size="small" color="#fff" />
          ) : (
            <Text style={styles.buttonText}>Scan Ticket</Text>
          )}
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.buttonWide}
          onPress={handleReviewTickets}
          disabled={isLoading}>
          {isFetching ? (
            <ActivityIndicator size="small" color="#fff" />
          ) : (
            <Text style={styles.buttonText}>Review Tickets</Text>
          )}
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}