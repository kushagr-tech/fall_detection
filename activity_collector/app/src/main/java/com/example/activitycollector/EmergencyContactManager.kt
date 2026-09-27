package com.example.activitycollector

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import org.json.JSONArray

/**
 * EmergencyContactManager
 * =======================
 * Manages storage and retrieval of emergency contacts in SharedPreferences.
 */
class EmergencyContactManager(context: Context) {

    companion object {
        private const val TAG = "EmergencyContactManager"
        private const val PREFS_NAME = "emergency_contacts_prefs"
        private const val KEY_CONTACTS = "key_contacts_list"
    }

    private val prefs: SharedPreferences =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    /**
     * Retrieve all saved emergency contacts in ordered sequence.
     */
    fun getContacts(): List<EmergencyContact> {
        val jsonStr = prefs.getString(KEY_CONTACTS, null) ?: return emptyList()
        val list = mutableListOf<EmergencyContact>()
        try {
            val jsonArray = JSONArray(jsonStr)
            for (i in 0 until jsonArray.length()) {
                val obj = jsonArray.getJSONObject(i)
                list.add(EmergencyContact.fromJson(obj))
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error parsing saved emergency contacts: ${e.message}")
        }
        return list
    }

    /**
     * Add a new emergency contact. Returns true if successfully saved.
     */
    fun addContact(name: String, phone: String): Boolean {
        val cleanName = name.trim()
        val cleanPhone = phone.trim().replace(" ", "").replace("-", "")
        if (cleanName.isEmpty() || cleanPhone.isEmpty()) return false

        val currentList = getContacts().toMutableList()
        currentList.add(EmergencyContact(name = cleanName, phone = cleanPhone))
        return saveContacts(currentList)
    }

    /**
     * Remove an emergency contact by ID.
     */
    fun removeContact(id: String): Boolean {
        val currentList = getContacts().toMutableList()
        val removed = currentList.removeAll { it.id == id }
        if (removed) {
            saveContacts(currentList)
        }
        return removed
    }

    /**
     * Returns the 1st contact (Primary), which receives the direct phone call.
     */
    fun getPrimaryContact(): EmergencyContact? {
        return getContacts().firstOrNull()
    }

    private fun saveContacts(contacts: List<EmergencyContact>): Boolean {
        return try {
            val jsonArray = JSONArray()
            for (c in contacts) {
                jsonArray.put(c.toJson())
            }
            prefs.edit().putString(KEY_CONTACTS, jsonArray.toString()).apply()
            true
        } catch (e: Exception) {
            Log.e(TAG, "Error saving emergency contacts: ${e.message}")
            false
        }
    }
}
