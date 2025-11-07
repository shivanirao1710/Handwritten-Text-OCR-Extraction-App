import React from 'react';
import { View, Text } from 'react-native';
import { styles } from '../styles/styles';

interface Props {
  label: string;
  value: string | number | null | undefined;
}

const DataRow: React.FC<Props> = ({ label, value }) => {
  if (!value) return null; // Don't render empty rows

  return (
    <View style={styles.dataRow}>
      <Text style={styles.dataLabel}>{label}</Text>
      <Text style={styles.dataValue}>{value}</Text>
    </View>
  );
};

export default DataRow;