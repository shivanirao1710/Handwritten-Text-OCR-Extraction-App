import React, { useState } from 'react';
import {
  Text,
  TouchableOpacity,
  View,
  Alert,
  ScrollView,
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

type Props = NativeStackScreenProps<AppStackParamList, 'Review'>;

export default function ReviewScreen({ route, navigation }: Props) {
  // Get the tickets passed from the Dashboard
  const { tickets: initialTickets } = route.params;
  const { authToken } = useAuth();

  const [tickets, setTickets] = useState<Ticket[]>(initialTickets);
  const [isUpdating, setIsUpdating] = useState(false);

  // States for edit functionality
  const [editModalVisible, setEditModalVisible] = useState(false);
  const [editingTicket, setEditingTicket] = useState<Ticket | null>(null);

  // Function to handle text editing
  const handleEditText = (ticket: Ticket) => {
    setEditingTicket(ticket);
    setEditModalVisible(true);
  };

  // Function to save edited text
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

  // Function to cancel editing
  const handleCancelEdit = () => {
    setEditModalVisible(false);
    setEditingTicket(null);
  };

  return (
    <SafeAreaView style={styles.container}>
      <Text style={styles.title}>Review Tickets</Text>
      <ScrollView style={styles.ticketsContainer}>
        {tickets.length > 0 ? (
          tickets.map((ticket) => (
            <TicketCard
              key={ticket.id}
              ticket={ticket}
              onEdit={() => handleEditText(ticket)}
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

      {/* Edit Modal */}
      {editingTicket && (
        <EditTicketModal
          visible={editModalVisible}
          ticket={editingTicket}
          isUpdating={isUpdating}
          onSave={handleSaveEditedText}
          onClose={handleCancelEdit}
        />
      )}
    </SafeAreaView>
  );
}