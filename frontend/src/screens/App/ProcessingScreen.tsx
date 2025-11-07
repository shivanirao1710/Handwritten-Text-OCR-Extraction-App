import React from 'react';
import { View, Text, ActivityIndicator } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { styles } from '../../styles/styles';

export default function ProcessingScreen() {
  return (
    <SafeAreaView style={styles.container}>
      <Text style={styles.title}>Processing...</Text>
      <View style={styles.processingContainer}>
        <ActivityIndicator size="large" color="#007bff" />
        <Text style={styles.processingText}>
          Uploading and scanning your ticket(s)
        </Text>
      </View>
    </SafeAreaView>
  );
}