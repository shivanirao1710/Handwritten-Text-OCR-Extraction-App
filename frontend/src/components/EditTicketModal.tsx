import React, { useState } from 'react';
import {
  Modal,
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ActivityIndicator,
} from 'react-native';
import { styles } from '../styles/styles';
import { Ticket } from '../types';

interface Props {
  visible: boolean;
  ticket: Ticket;
  isUpdating: boolean;
  onSave: (newRawText: string) => void;
  onClose: () => void;
}

const EditTicketModal: React.FC<Props> = ({
  visible,
  ticket,
  isUpdating,
  onSave,
  onClose,
}) => {
  // This component now manages the 'editedText' state locally
  const [editedText, setEditedText] = useState(ticket.raw_text_content);

  return (
    <Modal
      animationType="slide"
      transparent={true}
      visible={visible}
      onRequestClose={onClose}>
      <View style={styles.modalOverlay}>
        <View style={styles.modalContent}>
          <Text style={styles.modalTitle}>Edit Raw Text</Text>

          <TextInput
            style={styles.textInput}
            multiline
            numberOfLines={8}
            value={editedText}
            onChangeText={setEditedText}
            placeholder="Edit the raw extracted text here..."
          />

          <View style={styles.modalButtons}>
            <TouchableOpacity
              style={[styles.modalButton, styles.cancelButton]}
              onPress={onClose}
              disabled={isUpdating}>
              <Text style={styles.modalButtonText}>Cancel</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[styles.modalButton, styles.saveButton]}
              onPress={() => onSave(editedText)}
              disabled={isUpdating}>
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
  );
};

export default EditTicketModal;