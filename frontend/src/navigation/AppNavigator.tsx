import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { useAuth } from '../context/AuthContext';
import { Ticket } from '../types';

// Auth Screens
import LoginScreen from '../screens/Auth/LoginScreen';
import RegisterScreen from '../screens/Auth/RegisterScreen';

// App Screens
import DashboardScreen from '../screens/App/DashboardScreen';
import ProcessingScreen from '../screens/App/ProcessingScreen';
import ReviewScreen from '../screens/App/ReviewScreen';
import LoadingScreen from '../screens/LoadingScreen';

// Define navigation params
export type AuthStackParamList = {
  Login: undefined;
  Register: undefined;
};

export type AppStackParamList = {
  Dashboard: undefined;
  Processing: undefined;
  Review: { tickets: Ticket[] }; // We pass the list of tickets to this screen
};

const AuthStack = createNativeStackNavigator<AuthStackParamList>();
const AppStack = createNativeStackNavigator<AppStackParamList>();

const AppNavigator = () => {
  const { authToken, isLoading } = useAuth();

  if (isLoading) {
    // We can show the auth loading spinner, or a dedicated loading screen
    return <LoadingScreen />;
  }

  return (
    <NavigationContainer>
      {authToken ? (
        // User is logged in
        <AppStack.Navigator screenOptions={{ headerShown: false }}>
          <AppStack.Screen name="Dashboard" component={DashboardScreen} />
          <AppStack.Screen name="Processing" component={ProcessingScreen} />
          <AppStack.Screen name="Review" component={ReviewScreen} />
        </AppStack.Navigator>
      ) : (
        // User is not logged in
        <AuthStack.Navigator screenOptions={{ headerShown: false }}>
          <AuthStack.Screen name="Login" component={LoginScreen} />
          <AuthStack.Screen name="Register" component={RegisterScreen} />
        </AuthStack.Navigator>
      )}
    </NavigationContainer>
  );
};

export default AppNavigator;