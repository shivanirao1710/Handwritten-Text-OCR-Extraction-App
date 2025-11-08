// components/TicketCard.tsx

import React from 'react';
// REMOVED: Linking, Alert, API_BASE_URL
import { View, Text, TouchableOpacity } from 'react-native';
import { styles } from '../styles/styles';
import { Ticket } from '../types';
import DataRow from './DataRow';

interface Props {
  ticket: Ticket;
  onEdit: (ticket: Ticket) => void;
  onViewPdf: (ticket: Ticket) => void; // <-- ADD THIS PROP
}

const TicketCard: React.FC<Props> = ({ ticket, onEdit, onViewPdf }) => { // <-- Add prop to function
  
  // --- REMOVED the handleViewPdf function ---

  return (
    <View style={styles.ticket}>
      {/* --- RENDER PDF BUTTON --- */}
      <TouchableOpacity
        style={styles.editButton}
        onPress={() => onViewPdf(ticket)} // <-- UPDATE THIS OnPress
      >
        <Text style={styles.editButtonText}>View Original PDF</Text>
      </TouchableOpacity>

      {/* --- RENDER STRUCTURED DATA --- */}
      <View style={styles.structuredContainer}>
        <DataRow label="Ticket #" value={ticket.ticket_number} />
        <DataRow label="Date" value={ticket.ticket_date} />
        <DataRow label="Haul Vendor" value={ticket.haul_vendor} />
        <DataRow label="Truck #" value={ticket.truck_number} />
        <DataRow label="Material" value={ticket.material} />
        <DataRow label="Job #" value={ticket.job_number} />
        <DataRow label="Phase Code" value={ticket.phase_code} />
        <DataRow label="Zone" value={ticket.zone} />
        <DataRow label="Hours" value={ticket.hours} />
      </View>

      {/* --- RENDER RAW TEXT --- */}
      <Text style={styles.rawTextTitle}>Full Extracted Text:</Text>
      <Text style={styles.ticketText}>{ticket.raw_text_content}</Text>

      {/* --- Edit button --- */}
      <TouchableOpacity
        style={styles.editButton}
        onPress={() => onEdit(ticket)}>
        <Text style={styles.editButtonText}>Edit Raw Text</Text>
      </TouchableOpacity>
    </View>
  );
};

export default TicketCard;