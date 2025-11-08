// screens/ReviewScreen.tsx

import React, { useState } from 'react';
import {
  Text,
  TouchableOpacity,
  View,
  Alert,
  ScrollView,
  StyleSheet,
  Dimensions,
  ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useAuth } from '../../context/AuthContext';
import { styles } from '../../styles/styles';
import { Ticket } from '../../types';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { AppStackParamList } from '../../navigation/AppNavigator';
import TicketCard from '../../components/TicketCard';
import EditTicketModal from '../../components/EditTicketModal';
import { API_BASE_URL } from '../../api/config';
import Pdf from 'react-native-pdf';

type Props = NativeStackScreenProps<AppStackParamList, 'Review'>;

export default function ReviewScreen({ route, navigation }: Props) {
  // --- (Existing states are unchanged) ---
  const { tickets: initialTickets } = route.params;
  const { authToken } = useAuth(); // Make sure authToken is available
  const [tickets, setTickets] = useState<Ticket[]>(initialTickets);
  const [isUpdating, setIsUpdating] = useState(false);
  const [editModalVisible, setEditModalVisible] = useState(false);
  const [editingTicket, setEditingTicket] = useState<Ticket | null>(null);

  // --- (PDF states are unchanged) ---
  const [pdfSource, setPdfSource] = useState<object | null>(null);
  const [isPdfLoading, setIsPdfLoading] = useState<boolean>(false);

  // --- (handleEditText is unchanged) ---
  const handleEditText = (ticket: Ticket) => {
    setEditingTicket(ticket);
    setEditModalVisible(true);
  };

  // --- FIX 1: FILL IN THIS FUNCTION ---
  const handleSaveEditedText = async (newRawText: string) => {
    if (!editingTicket) return;

    setIsUpdating(true);
    try {
      const response = await fetch(`${API_BASE_URL}/update-ticket-text`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${authToken}`,
        },
        body: JSON.stringify({
          ticket_id: editingTicket.id,
          raw_text: newRawText,
        }),
      });

      if (response.ok) {
        const result = await response.json();
        const updatedTicket: Ticket = result.ticket;

        // Replace the entire old ticket with the new, re-parsed ticket
        const updatedTickets = tickets.map((ticket) =>
          ticket.id === editingTicket.id ? updatedTicket : ticket,
        );
        setTickets(updatedTickets); // Update the local state

        Alert.alert('Success', 'Text updated successfully!');
        setEditModalVisible(false);
        setEditingTicket(null);
      } else {
        const errorData = await response.json();
        Alert.alert(
          'Update Failed',
          errorData.detail || 'Failed to update text',
        );
      }
    } catch (error) {
      console.error('Update Error:', error);
      Alert.alert('Error', 'An error occurred while updating the text');
    } finally {
      setIsUpdating(false);
    }
  };

  // --- FIX 2: FILL IN THIS FUNCTION ---
  const handleCancelEdit = () => {
    setEditModalVisible(false);
    setEditingTicket(null);
  };

  // --- (PDF handlers are unchanged) ---
  const handleViewPdf = (ticket: Ticket) => {
    if (!ticket.pdf_url) {
      Alert.alert('Error', 'No PDF URL found for this ticket.');
      return;
    }
    const url = `${API_BASE_URL}${ticket.pdf_url}`;
    console.log('Loading PDF from:', url);

    setPdfSource({
      uri: url,
      cache: true,
      headers: {
        'ngrok-skip-browser-warning': 'true',
      },
    });
    setIsPdfLoading(true);
  };

  const handleClosePdf = () => {
    setPdfSource(null);
    setIsPdfLoading(false);
  };
  // ----------------------------------------------

  return (
    <SafeAreaView style={styles.container}>
      {/* ... (rest of your render logic is unchanged) ... */}
      <Text style={styles.title}>Review Tickets</Text>
      <ScrollView style={styles.ticketsContainer}>
        {tickets.length > 0 ? (
          tickets.map((ticket) => (
            <TicketCard
              key={ticket.id}
              ticket={ticket}
              onEdit={() => handleEditText(ticket)}
              onViewPdf={handleViewPdf}
            />
          ))
        ) : (
          <Text style={styles.noTicketsText}>No tickets found.</Text>
        )}
      </ScrollView>

      <TouchableOpacity
        style={styles.buttonWide}
        onPress={() => navigation.navigate('Dashboard')}>
        <Text style={styles.buttonText}>Back to Dashboard</Text>
      </TouchableOpacity>

      {/* --- (Edit Modal is unchanged) --- */}
      {editingTicket && (
        <EditTicketModal
          visible={editModalVisible}
          ticket={editingTicket}
          isUpdating={isUpdating}
          onSave={handleSaveEditedText}
          onClose={handleCancelEdit}
        />
      )}

      {/* --- (PDF Overlay is unchanged) --- */}
      {pdfSource && (
        <View style={pdfStyles.pdfViewerOverlay}>
          <Pdf
            source={pdfSource}
            trustAllCerts={false}
            style={pdfStyles.pdf}
            onLoadComplete={() => setIsPdfLoading(false)}
            onError={(error) => {
              console.log('PDF load error:', error);
              Alert.alert('Error', 'Failed to load PDF.');
              handleClosePdf();
            }}
          />
          <TouchableOpacity
            onPress={handleClosePdf}
            style={pdfStyles.closeButton}>
            <Text style={pdfStyles.closeButtonText}>✕ Close</Text>
          </TouchableOpacity>
          {isPdfLoading && (
            <ActivityIndicator
              size="large"
              color="#007BFF"
              style={pdfStyles.loadingIndicator}
            />
          )}
        </View>
      )}
    </SafeAreaView>
  );
}

// --- (PDF Styles are unchanged) ---
const pdfStyles = StyleSheet.create({
  pdfViewerOverlay: {
    position: 'absolute',
    zIndex: 1000,
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'white',
  },
  pdf: {
    flex: 1,
    width: Dimensions.get('window').width,
    height: Dimensions.get('window').height,
  },
  closeButton: {
    position: 'absolute',
    top: 50, // Adjust for status bar
    right: 20,
    zIndex: 1020,
    backgroundColor: 'rgba(0,0,0,0.5)',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 20,
  },
  closeButtonText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 16,
  },
  loadingIndicator: {
    position: 'absolute',
    top: '50%',
    left: '50%',
    zIndex: 1010,
  },
});