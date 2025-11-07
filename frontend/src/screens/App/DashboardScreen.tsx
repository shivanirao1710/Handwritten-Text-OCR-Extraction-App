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

        let successCount = 0;
        let failedCount = 0;

        for (const [index, imageUri] of scannedImages.entries()) {
          console.log(
            `Processing image ${index + 1} of ${scannedImages.length}...`,
          );
          try {
            const formData = new FormData();
            formData.append('file', {
              uri: imageUri,
              type: 'image/jpeg',
              name: `ticket_page_${index + 1}.jpg`,
            });

            const response = await fetch(`${API_BASE_URL}/scan`, {
              method: 'POST',
              headers: { Authorization: `Bearer ${authToken}` },
              body: formData,
            });

            if (response.ok) {
              successCount++;
            } else {
              const errorData = await response.json();
              console.error(
                `Failed to process image ${index + 1}:`,
                errorData.detail,
              );
              failedCount++;
            }
          } catch (uploadError) {
            console.error(`Error uploading image ${index + 1}:`, uploadError);
            failedCount++;
          }
        }

        Alert.alert(
          'Processing Complete',
          `Successfully processed ${successCount} image(s).\nFailed to process ${failedCount} image(s).`,
        );

        // After scanning, automatically fetch the new list and go to review
        if (successCount > 0) {
          // Re-fetch the full list to include the new tickets
          try {
            const response = await fetch(`${API_BASE_URL}/tickets`, {
              method: 'GET',
              headers: { Authorization: `Bearer ${authToken}` },
            });
            if (response.ok) {
              const data: Ticket[] = await response.json();
              navigation.navigate('Review', { tickets: data }); // Navigate with fresh data
            } else {
              Alert.alert('Fetch Failed', 'Could not retrieve new tickets.');
              navigation.navigate('Dashboard'); // Go back to dashboard
            }
          } catch {
            Alert.alert(
              'Error',
              'An error occurred while fetching new tickets.',
            );
            navigation.navigate('Dashboard');
          }
        } else {
          navigation.navigate('Dashboard'); // Go back to dashboard if all failed
        }
      } else {
        setIsScanning(false); // No images scanned
      }
    } catch (error) {
      console.error('Scan or Upload Error:', error);
      Alert.alert(
        'Error',
        'An error occurred during the scan. Please try again.',
      );
      navigation.navigate('Dashboard');
    }
    // Note: We don't set isScanning(false) here because navigation is handling the screen change.
    // It will be false when the user returns to this screen.
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